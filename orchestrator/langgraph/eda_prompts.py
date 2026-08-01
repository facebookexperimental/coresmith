# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deployment-fed prompt context + rollback-flag prompt selection (PR4).

The autonomous-EDA nodes (``backend_graph._run_llm_eda_step`` and
``tapeout_graph._run_tapeout_llm_step``) hand a ``*_llm.md`` / ``tapeout_wrapper_*.md``
prompt to a Bash-capable LLM. Post-migration those prompts:

* run each EDA verb through the coresmith CLI (``"$CS" tool run_<verb> ...``)
  instead of a bare ``yosys``/``openroad``/``magic``/``netgen`` invocation, and
* get their tool/PDK-specific recipe text from the active
  :class:`~orchestrator.pdk.base.Deployment` (``{tool_notes}`` / ``{pdk_summary}``)
  rather than a hardcoded Sky130 block.

This module owns the two seams that make that work for BOTH graphs:

1. :func:`deployment_prompt_context` -- the fields to MERGE into a node's
   context dict (the caller's keys win on collision), including a verb-specific
   ``tool_notes`` resolved from that verb's :meth:`EdaTool.prompt_notes`.
2. :func:`resolve_prompt_path` -- honours ``CORESMITH_TOOL_CLI_PROMPTS``
   (default ON): when OFF, a ``<name>.legacy.md`` sibling (the pre-migration
   text) is selected instead, so an in-flight run can be rolled back mid-flight.
"""

from __future__ import annotations

from pathlib import Path

# Which verb's prompt_notes() each migrated prompt should receive as
# ``{tool_notes}``. Prompts absent here (e.g. backend_wrapper_llm.md, which only
# packages already-produced artifacts) get an empty ``tool_notes``.
_PROMPT_VERB: dict[str, str] = {
    "backend_synth_llm.md": "run_synth",
    "backend_pnr_llm.md": "run_pnr",
    "backend_drc_llm.md": "run_drc",
    "backend_lvs_llm.md": "run_lvs",
    "tapeout_wrapper_synth.md": "run_synth",
    "tapeout_wrapper_pnr.md": "run_pnr",
    "tapeout_wrapper_drc.md": "run_drc",
    "tapeout_wrapper_lvs.md": "run_lvs",
}


def tool_cli_prompts_enabled() -> bool:
    """Are the CLI-based (migrated) prompts active? Default ON.

    Set ``CORESMITH_TOOL_CLI_PROMPTS=0`` to fall back to the pre-migration
    ``.legacy.md`` prompt text (cheap, honest rollback for a release cycle).
    """
    try:
        from orchestrator.profile import ensure_applied, flag_enabled
        ensure_applied()
        return flag_enabled("CORESMITH_TOOL_CLI_PROMPTS", default=True)
    except Exception:  # noqa: BLE001 - a profile hiccup must not break the node
        return True


def resolve_prompt_path(prompt_dir: Path, prompt_file: str) -> Path:
    """Return the prompt file to use, honoring the rollback flag.

    When ``CORESMITH_TOOL_CLI_PROMPTS`` is OFF and a ``<stem>.legacy.md`` sibling
    exists, that pre-migration copy is used; otherwise the (migrated) prompt.
    """
    prompt_dir = Path(prompt_dir)
    if not tool_cli_prompts_enabled():
        legacy = prompt_dir / f"{Path(prompt_file).stem}.legacy.md"
        if legacy.is_file():
            return legacy
    return prompt_dir / prompt_file


def deployment_prompt_context(prompt_file: str) -> dict[str, str]:
    """Deployment-supplied fields to merge into a prompt's context dict.

    Best-effort: any resolution error yields ``{}`` so a node never crashes on
    the deployment layer. ``tool_notes`` is the prompt's verb-specific
    :meth:`EdaTool.prompt_notes` (empty when the prompt maps to no verb).
    """
    try:
        from orchestrator.pdk.registry import get_deployment

        dep = get_deployment()
        ctx: dict[str, str] = {str(k): str(v)
                               for k, v in dep.prompt_context().items()}
        verb = _PROMPT_VERB.get(prompt_file)
        notes = ""
        if verb and dep.supports(verb):
            try:
                notes = dep.tool(verb).prompt_notes() or ""
            except Exception:  # noqa: BLE001
                notes = ""
        ctx["tool_notes"] = notes
        ctx.setdefault("pdk_summary", dep.name)
        return ctx
    except Exception:  # noqa: BLE001
        return {}


def merged_prompt_context(prompt_file: str, context: dict) -> dict:
    """``deployment_prompt_context()`` overlaid by the caller's ``context``
    (caller keys win on collision), as the migration wiring merges them."""
    merged = deployment_prompt_context(prompt_file)
    merged.update(context)
    return merged
