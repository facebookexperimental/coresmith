# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
UarchSpecGenerator -- Agent that produces a microarchitecture spec.

Reads a Python golden model, understands the algorithm, and generates
a detailed microarchitecture specification that an RTL engineer (or the
RTL generator LLM) can implement unambiguously.

The spec covers interfaces, datapath, control FSM, storage elements,
algorithm mapping, timing, and edge cases -- every decision needed
before writing Verilog.
"""

from __future__ import annotations

from orchestrator._timeouts import scaled
import json
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .socmate_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "uarch_spec_generator.md"
)
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = (
        "You are an expert digital VLSI micro-architect. "
        "Produce a detailed microarchitecture specification from a Python model."
    )


class UarchSpecGenerator:
    """Agent for generating microarchitecture specifications."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from orchestrator.langchain.agents.socmate_llm import block_model
        model = model or block_model()
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(2700, env="SOCMATE_UARCH_TIMEOUT"),
        )

    async def generate(
        self,
        block_name: str,
        python_source: str,
        description: str = "",
        feedback: str = "",
        previous_spec: str = "",
        constraints: list[dict] | None = None,
        callbacks: list | None = None,
        project_root: str = "",
    ) -> dict[str, Any]:
        """Generate a microarchitecture specification from Python source.

        Args:
            block_name: Name of the hardware block.
            python_source: Python source code of the golden model.
            description: Human-readable description of the block.
            feedback: Human feedback for revision (if revising a prior spec).
            previous_spec: The prior spec text being revised.
            constraints: Accumulated design constraints.
            callbacks: Optional callbacks (unused, kept for API compat).
            project_root: Path to project root for reading arch docs from disk.

        Returns:
            Dict with keys: spec_text, spec_summary, block_name
        """
        block_title = block_name.replace("_", " ").title()
        revision_label = " - Revision" if previous_spec else ""
        span_name = f"Uarch Spec [{block_title}]{revision_label}"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("is_revision", bool(previous_spec))

            parts = [
                "Generate a microarchitecture specification for the following block.",
                f"\nBlock name: {block_name}",
                f"Description: {description}",
            ]

            if constraints:
                parts.append("\n--- DESIGN CONSTRAINTS ---")
                for i, c in enumerate(constraints, 1):
                    if isinstance(c, dict):
                        parts.append(f"  {i}. {c.get('rule', str(c))}")
                    else:
                        parts.append(f"  {i}. {c}")
                parts.append("")

            if previous_spec and feedback:
                parts.append(
                    "\n--- REVISION REQUESTED ---\n"
                    f"The previous specification was reviewed and needs changes.\n"
                    f"Human feedback:\n{feedback}\n"
                    f"\n--- Previous Specification ---\n{previous_spec}\n"
                    f"\nRevise the specification to address ALL feedback points.\n"
                )
            elif previous_spec:
                parts.append(
                    "\n--- REVISION REQUESTED ---\n"
                    "The previous specification was rejected. Please revise it.\n"
                    f"\n--- Previous Specification ---\n{previous_spec}\n"
                )

            # Read architecture docs from disk so the LLM actually has
            # the ERS, block diagram connections, and FRD it's told to follow
            if project_root:
                from pathlib import Path as _P
                import json as _json

                _root = _P(project_root)

                # ERS -- the authoritative engineering spec
                ers_path = _root / "arch" / "ers_spec.md"
                if ers_path.exists():
                    try:
                        parts.append(
                            "\n--- ENGINEERING REQUIREMENTS SPECIFICATION (ERS) ---\n"
                            f"{ers_path.read_text()}\n"
                            "--- END ERS ---\n"
                        )
                    except OSError:
                        pass

                # Block diagram connections for this block
                bd_path = _root / ".socmate" / "block_diagram.json"
                if bd_path.exists():
                    try:
                        bd = _json.loads(bd_path.read_text())
                        conns = [
                            c for c in bd.get("connections", [])
                            if c.get("from") == block_name or c.get("to") == block_name
                        ]
                        if conns:
                            parts.append(
                                "\n--- CONNECTION GRAPH (this block's connections) ---\n"
                                f"{_json.dumps(conns, indent=2)}\n"
                                "--- END CONNECTION GRAPH ---\n"
                            )

                        for blk in bd.get("blocks", []):
                            if blk.get("name") == block_name:
                                ifaces = blk.get("interfaces", {})
                                if ifaces:
                                    parts.append(
                                        "\n--- BLOCK INTERFACES (from block diagram) ---\n"
                                        f"{_json.dumps(ifaces, indent=2)}\n"
                                    )
                                break
                    except (OSError, _json.JSONDecodeError):
                        pass

                # FRD for testable requirements context
                frd_path = _root / "arch" / "frd_spec.md"
                if frd_path.exists():
                    try:
                        frd_text = frd_path.read_text()
                        if len(frd_text) > 8000:
                            frd_text = frd_text[:8000] + "\n... (truncated)"
                        parts.append(
                            "\n--- FUNCTIONAL REQUIREMENTS (FRD) ---\n"
                            f"{frd_text}\n"
                            "--- END FRD ---\n"
                        )
                    except OSError:
                        pass

            parts.append(
                f"\n--- Python Golden Model ---\n```python\n{python_source}\n```"
            )

            user_message = "\n".join(parts)

            run_name = f"Generate Uarch Spec [{block_title}]{revision_label}"
            content = await self.llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name=run_name,
            )

            spec_text, spec_summary = self._parse_response(content, block_name)

            return {
                "spec_text": spec_text,
                "spec_summary": spec_summary,
                "block_name": block_name,
            }

    def _parse_response(
        self, content: str, block_name: str
    ) -> tuple[str, dict]:
        """Extract the spec document and JSON summary from the LLM response."""
        spec_text = content.strip()

        # Extract JSON summary block
        spec_summary = {}
        json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_match:
            try:
                spec_summary = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        if not spec_summary:
            spec_summary = {"block_name": block_name}

        return spec_text, spec_summary
