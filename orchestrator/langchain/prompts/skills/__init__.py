"""Skill loader for agent system prompts.

Skills are reference markdown documents under
`orchestrator/langchain/prompts/skills/`. Agents that author or consume
hardware interfaces should pull the relevant skills into their system
prompt so they have access to coresmith's interface conventions.

Usage:
    from orchestrator.langchain.prompts.skills import load_skill

    sys_prompt = base_prompt + "\\n\\n" + load_skill("axi_stream")

Each skill is a markdown file named `<id>.md`. The top-level heading
and surrounding text become the prompt fragment as-is.
"""
from __future__ import annotations

from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent


def load_skill(skill_id: str) -> str:
    """Return the markdown body of a skill, or empty string if missing.

    `skill_id` matches the filename stem (e.g. `"axi_stream"` for
    `axi_stream.md`). Missing skills return empty rather than raising
    so a prompt builder degrades gracefully.
    """
    path = _SKILLS_DIR / f"{skill_id}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_skills(*skill_ids: str, separator: str = "\n\n---\n\n") -> str:
    """Concatenate multiple skills for inclusion in a system prompt.

    Empty/missing skills are skipped so adding a new skill id never
    breaks an agent that hasn't shipped it yet.
    """
    parts = [load_skill(sid) for sid in skill_ids]
    return separator.join(p for p in parts if p)


__all__ = ["load_skill", "load_skills"]
