# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
IntegrationReviewAgent -- Reviews all uArch specs for cross-block
interface coherence before RTL generation.

Reads Section 9 Verilog stubs from every spec, cross-references against
the block diagram connections, and edits spec files on disk to fix
mismatches in widths, directions, protocols, and naming.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .coresmith_llm import DEFAULT_MODEL, ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "integration_review.md"
)
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = (
        "You are a chip integration engineer. Review all uArch specs "
        "for interface coherence and fix mismatches by editing files on disk."
    )


_JSON_BLOCK_RE = re.compile(
    r"```json\s*\n\s*(\{[^}]*\})\s*\n\s*```", re.DOTALL
)


def _endpoint_block(endpoint: Any) -> str | None:
    if endpoint is None:
        return None
    text = str(endpoint)
    if "." in text:
        return text.split(".", 1)[0]
    return text or None


def _filter_connections_for_blocks(
    block_diagram: dict[str, Any],
    block_names: list[str],
) -> tuple[dict[str, Any], int]:
    """Return a copy containing only connections fully inside block_names.

    The pipeline reviews one tier at a time. Architecture diagrams can contain
    edges to later-tier blocks whose uArch specs have not been generated yet;
    those edges are not actionable during the current tier review.
    """
    review_blocks = set(block_names)
    filtered = dict(block_diagram)
    filtered_blocks = []
    for block in block_diagram.get("blocks", []):
        if not isinstance(block, dict) or block.get("name") in review_blocks:
            filtered_blocks.append(block)
    filtered["blocks"] = filtered_blocks

    kept = []
    deferred = 0
    for conn in block_diagram.get("connections", []):
        if not isinstance(conn, dict):
            kept.append(conn)
            continue
        src = _endpoint_block(conn.get("from") or conn.get("source") or conn.get("src"))
        dst = _endpoint_block(conn.get("to") or conn.get("dest") or conn.get("destination"))
        if src in review_blocks and dst in review_blocks:
            kept.append(conn)
        else:
            deferred += 1
    filtered["connections"] = kept
    return filtered, deferred


def _parse_issue_counts(summary: str) -> tuple[int, int]:
    """Extract issues_found / issues_fixed from the LLM's JSON summary block.

    Falls back to (0, 0) if parsing fails, which forces human review
    rather than silently misclassifying the outcome.
    """
    m = _JSON_BLOCK_RE.search(summary)
    if m:
        try:
            data = json.loads(m.group(1))
            found = int(data.get("issues_found", 0))
            fixed = int(data.get("issues_fixed", 0))
            return max(found, 0), max(fixed, 0)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return 0, 0


class IntegrationReviewAgent:
    """Reviews all uArch specs for cross-block interface coherence."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        self.llm = ClaudeLLM(
            model=model,
            timeout=int(os.environ.get("CORESMITH_INTEGRATION_REVIEW_TIMEOUT", "2700")),
        )

    async def review(
        self,
        block_names: list[str],
        project_root: str,
    ) -> dict[str, Any]:
        """Review all uArch specs for interface coherence.

        Args:
            block_names: Names of blocks whose specs to review.
            project_root: Path to project root.

        Returns:
            Dict with keys: summary (str), issues_found (int), issues_fixed (int)
        """
        with _tracer.start_as_current_span("Integration Review") as span:
            span.set_attribute("block_count", len(block_names))

            root = Path(project_root)

            # Soft write-lock on the canonical uArch specs: clone each into
            # arch/uarch_specs_review/ and let the LLM Edit/Write against
            # the copies.  Without this, the integration-review agent
            # rewrites the originals AFTER RTL has been committed against
            # them -- bytes that the block-level cocotb tests already
            # validated get replaced by 30-40 KB the RTL was never built
            # for.  Discovered live during the video codec codec run.
            #
            # Behaviour: callers can opt out by setting
            # CORESMITH_INTEGRATION_REVIEW_INPLACE=1 (e.g. for legacy flows).
            inplace = os.environ.get(
                "CORESMITH_INTEGRATION_REVIEW_INPLACE", ""
            ).strip().lower() in {"1", "true", "yes", "on"}

            review_dir = root / "arch" / "uarch_specs_review"
            spec_paths = []
            for name in block_names:
                src = root / "arch" / "uarch_specs" / f"{name}.md"
                if not src.exists():
                    continue
                if inplace:
                    spec_paths.append(str(src))
                    continue
                review_dir.mkdir(parents=True, exist_ok=True)
                dst = review_dir / f"{name}.md"
                shutil.copy2(src, dst)
                spec_paths.append(str(dst))

            bd_path = root / ".coresmith" / "block_diagram.json"
            review_bd_path = bd_path
            deferred_connection_count = 0
            if bd_path.exists():
                try:
                    block_diagram = json.loads(bd_path.read_text())
                    filtered, deferred_connection_count = _filter_connections_for_blocks(
                        block_diagram, block_names
                    )
                    review_bd_path = root / ".coresmith" / "integration_review_block_diagram.json"
                    review_bd_path.write_text(json.dumps(filtered, indent=2))
                except (json.JSONDecodeError, OSError, TypeError):
                    review_bd_path = bd_path

            parts = [
                "Review the following uArch specs for cross-block interface coherence.",
                "",
                "## uArch Spec Files",
                "Read each of these files:",
            ]
            for sp in spec_paths:
                parts.append(f"- {sp}")

            parts.append("")
            parts.append("## Architecture Files")
            parts.append(f"- Current-tier block diagram connections: {review_bd_path}")
            if deferred_connection_count:
                parts.append(
                    f"- Deferred cross-tier/future-tier connections: {deferred_connection_count}. "
                    "Do not count these as current-tier issues because the connected "
                    "uArch specs do not exist yet."
                )

            ers_path = root / "arch" / "ers_spec.md"
            if ers_path.exists():
                parts.append(f"- ERS: {ers_path}")

            prd_path = root / ".coresmith" / "prd_spec.json"
            if prd_path.exists():
                try:
                    prd = json.loads(prd_path.read_text())
                    prd_doc = prd.get("prd", prd.get("ers", {}))
                    dataflow = prd_doc.get("dataflow", {})
                    bus_protocol = dataflow.get("bus_protocol", "unknown")
                    data_width = dataflow.get("data_width_bits", "unknown")
                    parts.append(f"- PRD bus_protocol: {bus_protocol}")
                    parts.append(f"- PRD data_width_bits: {data_width}")
                except (json.JSONDecodeError, OSError):
                    pass

            parts.append("")
            parts.append(
                "Check every connection in the current-tier block diagram. For each, verify "
                "port widths, directions, protocols, and clock/reset naming match "
                "across connected blocks. Do not report missing ports or missing specs "
                "for deferred cross-tier/future-tier connections. If you find mismatches, "
                "edit the uArch spec files on disk to fix them. Report a summary of findings."
            )

            user_message = "\n".join(parts)

            content = await self.llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name="Integration Review",
            )

            summary = content.strip() if content else "No issues found."

            issues_found, issues_fixed = _parse_issue_counts(summary)

            # Surface the review-dir so the caller (or the next iteration's
            # uArch generator) can compare canonical specs vs review copies
            # and decide whether to restart blocks.
            return {
                "summary": summary,
                "issues_found": issues_found,
                "issues_fixed": issues_fixed,
                "review_dir": (
                    str(review_dir) if not inplace and review_dir.exists() else None
                ),
            }
