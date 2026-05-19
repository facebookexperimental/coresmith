# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Block Diagram Specialist -- proposes or refines block-level ASIC architecture.

Standalone async function that can be called directly via MCP tools or
wrapped as a LangGraph node in Phase 3.

Uses ClaudeLLM for LLM inference. Takes requirements, PDK summary,
benchmark data, and constraint feedback to produce a structured block
diagram with AXI-Stream interfaces.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "block_diagram.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


def _scan_golden_models(
    project_root: str = ".",
    model_dirs: list[dict] | None = None,
    requirements: str = "",
) -> str:
    """Scan Python golden model directories and extract interface signatures.

    Fix #1: Now accepts *model_dirs* (list of ``{"path": ..., "label": ...}``
    dicts) so different projects can specify their own source directories via
    ``config.yaml`` ``golden_model_dirs``.

    Falls back to auto-discovery (top-level subdirectories containing
    ``*.py`` with class definitions, max depth 2) when no dirs are
    specified.

    Returns:
        A formatted string summarising the discovered models.
    """
    from pathlib import Path
    import ast

    roots: list[Path] = []
    for candidate in (
        Path(project_root),
        Path(os.environ.get("SOCMATE_SOURCE_ROOT", "") or "."),
        Path(__file__).resolve().parents[3],
    ):
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    if not roots:
        roots.append(Path(__file__).resolve().parents[3])

    # Resolve configured dirs
    if model_dirs:
        scan_dirs = [(d["path"], d.get("label", d["path"])) for d in model_dirs]
    else:
        # Try loading from config.yaml
        scan_dirs = []
        for root in roots:
            scan_dirs = _load_golden_model_dirs_from_config(root)
            if scan_dirs:
                break

    if not scan_dirs:
        # Auto-discover: scan top-level subdirectories (depth <= 2) for
        # Python files that contain class definitions.
        scan_dirs = _auto_discover_model_dirs(roots[0])

    referenced_paths = sorted(set(re.findall(r"[\w./-]+\.py", requirements)))

    summaries: list[str] = []
    seen_files: set[Path] = set()

    def summarize_file(py_file: Path, root: Path, subsystem: str) -> None:
        resolved = py_file.resolve()
        if resolved in seen_files or py_file.name.startswith("_") or not py_file.exists():
            return
        seen_files.add(resolved)
        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except Exception:
            return

        lines: list[str] = []
        module_doc = ast.get_docstring(tree)
        if module_doc:
            first = " ".join(module_doc.strip().split())[:500]
            lines.append(f"    module_doc: {first}")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not n.name.startswith("_")
                ]
                methods_str = ", ".join(methods[:8])
                if len(methods) > 8:
                    methods_str += f", ... (+{len(methods) - 8})"
                lines.append(f"    class {node.name}: [{methods_str}]")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                args = [arg.arg for arg in node.args.args]
                doc = ast.get_docstring(node)
                doc_part = f" -- {' '.join(doc.split())[:180]}" if doc else ""
                lines.append(f"    function {node.name}({', '.join(args)}){doc_part}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        lines.append(f"    constant {target.id}")

        if not lines:
            return
        try:
            rel_path = py_file.relative_to(root)
        except ValueError:
            rel_path = py_file
        summaries.append(f"  {rel_path} ({subsystem}):\n" + "\n".join(lines[:24]))

    for ref in referenced_paths:
        for root in roots:
            candidate = (root / ref).resolve()
            if candidate.exists() and candidate.is_file():
                summarize_file(candidate, root, "referenced golden model")
                break

    for rel_dir, subsystem in scan_dirs:
        for root in roots:
            src_dir = root / rel_dir
            if not src_dir.is_dir():
                continue
            for py_file in sorted(src_dir.glob("*.py")):
                summarize_file(py_file, root, subsystem)

    if not summaries:
        return ""

    return (
        "GOLDEN MODEL SOURCES (ground your block names, interfaces, and "
        "python_source paths on these actual files):\n"
        + "\n".join(summaries)
    )


def _load_golden_model_dirs_from_config(root: Path) -> list[tuple[str, str]]:
    """Try to read ``golden_model_dirs`` from config.yaml."""
    import yaml

    config_path = root / "orchestrator" / "config.yaml"
    if not config_path.exists():
        return []
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        entries = cfg.get("golden_model_dirs", [])
        return [(e["path"], e.get("label", e["path"])) for e in entries if "path" in e]
    except Exception:
        return []


def _auto_discover_model_dirs(root: Path) -> list[tuple[str, str]]:
    """Auto-discover directories containing Python golden models.

    Scans up to depth 2 for directories that contain ``*.py`` files with
    at least one class definition.  Skips hidden dirs, ``__pycache__``,
    ``venv``, ``node_modules``, and the ``orchestrator`` package itself.
    """
    import ast

    skip = {"__pycache__", ".git", "venv", ".venv", "node_modules",
            "orchestrator", "sim_build", "syn", ".socmate"}

    found: list[tuple[str, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in skip or child.name.startswith("."):
            continue
        # Check this dir and one level deeper
        for scan_dir in [child] + [d for d in child.iterdir() if d.is_dir() and d.name not in skip]:
            has_class = False
            for py_file in scan_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                try:
                    tree = ast.parse(py_file.read_text())
                    if any(isinstance(n, ast.ClassDef) for n in ast.walk(tree)):
                        has_class = True
                        break
                except Exception:
                    continue
            if has_class:
                try:
                    rel = scan_dir.relative_to(root)
                except ValueError:
                    continue
                label = str(rel).replace("/", " / ")
                found.append((str(rel), label))
    return found


async def analyze_block_diagram(
    requirements: str,
    pdk_summary: str,
    target_clock_mhz: float,
    existing_diagram: dict | None = None,
    constraint_feedback: list[str] | None = None,
    benchmark_data: dict | None = None,
    human_feedback: str = "",
    project_root: str = ".",
    ers_spec: dict | None = None,
) -> dict[str, Any]:
    """Propose or refine a block-level ASIC architecture.

    This is a standalone function -- no LangGraph or Temporal dependency.
    Can be called directly from MCP tools or wrapped as a graph node.

    Args:
        requirements: High-level spec text from the architect.
        pdk_summary: Human-readable PDK summary (from PDKConfig.to_summary()).
        target_clock_mhz: Target clock frequency.
        existing_diagram: Previous block diagram to refine (None for first pass).
        constraint_feedback: List of constraint violations to address.
        benchmark_data: Dict of benchmark results for gate count grounding.
        human_feedback: Architect's feedback from a previous iteration.
        project_root: Path to project root for golden model scanning.
        ers_spec: Product Requirements Document (structured dict from
            the PRD specialist).  Provides authoritative sizing data.

    Returns:
        Dict with keys: blocks, connections, reasoning, questions.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("socmate.architecture.block_diagram")

    with tracer.start_as_current_span("analyze_block_diagram") as span:
        span.set_attribute("target_clock_mhz", target_clock_mhz)
        span.set_attribute("has_existing_diagram", existing_diagram is not None)
        span.set_attribute("constraint_count", len(constraint_feedback or []))

        # Scan actual golden model sources for grounding
        golden_model_context = _scan_golden_models(project_root, requirements=requirements)

        # Read SAD and FRD from disk if available
        sad_context = ""
        frd_context = ""
        root = Path(project_root)
        sad_path = root / "arch" / "sad_spec.md"
        frd_path = root / "arch" / "frd_spec.md"
        if sad_path.exists():
            try:
                sad_context = sad_path.read_text()
            except OSError:
                pass
        if frd_path.exists():
            try:
                frd_context = frd_path.read_text()
            except OSError:
                pass

        # Build context sections for the prompt
        benchmark_context = ""
        if benchmark_data:
            benchmark_context = (
                "BENCHMARK DATA (use these for gate count estimates instead of guessing):\n"
                + json.dumps(benchmark_data, indent=2)
            )

        constraint_context = ""
        if constraint_feedback:
            constraint_context = (
                "CONSTRAINT VIOLATIONS from previous iteration (you must address these):\n"
                + "\n".join(f"  - {v}" for v in constraint_feedback)
                + "\n\nRepair requirements:\n"
                "- Revise the previous block diagram so every listed violation is either fixed or explicitly converted into a blocking question.\n"
                "- For throughput, latency, bandwidth, frame-rate, tile-rate, or cycles-per-transaction violations, include explicit arithmetic in system_invariants: clock cycles available, transaction count, cycles per transaction, and claimed latency/throughput.\n"
                "- Do not keep a stale latency/throughput number that conflicts with the PRD/FRD KPI. Prefer tightening the local pipeline contract over weakening the user KPI.\n"
                "- In reasoning, list each violation and the concrete before/after repair."
            )

        feedback_context = ""
        if human_feedback:
            feedback_context = (
                f"ARCHITECT FEEDBACK (incorporate this into your proposal):\n  {human_feedback}"
            )

        # Build the user message
        parts = [
            "Design the block-level architecture for the following ASIC.",
            f"\nRequirements: {requirements}",
            f"\nTarget process: {pdk_summary}",
            f"Target clock: {target_clock_mhz} MHz",
        ]

        if ers_spec:
            prd_doc = ers_spec.get("prd", ers_spec.get("ers", ers_spec))
            parts.append(
                f"\n--- PRODUCT REQUIREMENTS DOCUMENT (PRD) ---\n"
                f"{json.dumps(prd_doc, indent=2)}"
            )

        if sad_context:
            parts.append(
                f"\n--- SYSTEM ARCHITECTURE DOCUMENT (SAD) ---\n"
                f"{sad_context}"
            )

        if frd_context:
            parts.append(
                f"\n--- FUNCTIONAL REQUIREMENTS DOCUMENT (FRD) ---\n"
                f"{frd_context}"
            )

        if golden_model_context:
            parts.append(f"\n{golden_model_context}")

        if existing_diagram:
            parts.append(
                f"\n--- PREVIOUS BLOCK DIAGRAM (refine this) ---\n"
                f"{json.dumps(existing_diagram, indent=2)}"
            )

        target_path = Path(project_root) / ".socmate" / "block_diagram.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        parts.append(
            f"\nIMPORTANT: Write the block diagram JSON to: {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )

        user_message = "\n".join(parts)

        # Fill template variables in the system prompt
        system_prompt = SYSTEM_PROMPT.format(
            benchmark_context=benchmark_context,
            constraint_context=constraint_context,
            feedback_context=feedback_context,
        )

        # Import here to avoid circular deps and allow mocking in tests
        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="block_diagram",
            )
            from orchestrator.utils import read_back_json
            default = {
                "blocks": [],
                "connections": [],
                "reasoning": "",
                "questions": [],
            }
            disk_result, disk_ok = read_back_json(
                target_path, content, default, context="block_diagram"
            )
            result = disk_result if disk_ok else _parse_response(content)

            # Detect LLM errors returned as content (not exceptions).
            # ClaudeLLM swallows timeouts and returns error text as
            # string content, bypassing the except handler below.
            if not result["blocks"] and not result["questions"]:
                reasoning = result.get("reasoning", "")
                is_timeout = "timed out" in reasoning
                is_llm_error = "[ClaudeLLM error:" in reasoning
                suffix = (
                    " (LLM timeout — consider increasing timeout or checking API key)"
                    if is_timeout
                    else " (LLM returned non-JSON response)"
                    if is_llm_error
                    else ""
                )
                result["questions"].append({
                    "question": f"Block diagram generation failed{suffix}. "
                                "Please review requirements or retry.",
                    "context": reasoning[:500],
                    "priority": "blocking",
                })
                span.set_attribute("error_detected_in_content", True)

            span.set_attribute("block_count", len(result.get("blocks", [])))
            span.set_attribute("connection_count", len(result.get("connections", [])))
            span.set_attribute("question_count", len(result.get("questions", [])))
            return result

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(_trace.StatusCode.ERROR, str(e))
            return {
                "blocks": [],
                "connections": [],
                "reasoning": f"Block diagram agent error: {e}",
                "questions": [
                    {
                        "question": "Block diagram generation failed. Can you provide more details about the requirements?",
                        "context": f"Error: {e}",
                        "priority": "blocking",
                    }
                ],
            }


def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default = {
        "blocks": [],
        "connections": [],
        "reasoning": "",
        "questions": [],
    }
    result, _ok = parse_llm_json(content, default, context="block_diagram")
    return result
