# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Architecture state schema and persistence.

Defines the ArchitectureState dataclass (the shared state across all
specialist agents) and ArchitectureQuestion (the data model for agent-
initiated questions to the human architect).

State is persisted as JSON at .coresmith/architecture_state.json so that
MCP tool calls across a Claude CLI conversation share the same state.
In Phase 3, ArchitectureState can be converted to a LangGraph TypedDict
with message reducers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_DIR = ".coresmith"
ARCH_DOC_DIR = "arch"
STATE_FILE = "architecture_state.json"


# ---------------------------------------------------------------------------
# ArchitectureQuestion
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureQuestion:
    """A question from a specialist agent to the human architect.

    Attributes:
        id: Unique identifier (uuid4).
        agent: Which specialist asked (e.g. "block_diagram", "lead_architect").
        question: The question text.
        context: Why this matters for the architecture.
        options: Suggested answers, or None for free-form.
        priority: "blocking" (halts graph), "clarifying", or "nice_to_have".
        answer: Human's answer, once provided.
        timestamp: ISO 8601 when the question was created.
        answered_at: ISO 8601 when the answer was provided.
    """

    id: str = ""
    agent: str = ""
    question: str = ""
    context: str = ""
    options: list[str] | None = None
    priority: str = "clarifying"
    answer: str | None = None
    timestamp: str = ""
    answered_at: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def is_answered(self) -> bool:
        return self.answer is not None

    def provide_answer(self, answer: str) -> None:
        self.answer = answer
        self.answered_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureQuestion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ArchitectureState
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureState:
    """Shared state across all architecture specialist agents.

    This is a plain dataclass (not a LangGraph TypedDict) so it can be
    serialized to JSON on disk for persistence between MCP tool calls.
    In Phase 3, it can be converted to a TypedDict with reducers.
    """

    # --- Inputs ---
    requirements: str = ""
    pdk_config: dict = field(default_factory=dict)
    target_clock_mhz: float = 50.0
    human_feedback: str = ""

    # --- PRD (Product Requirements Document) ---
    prd_spec: dict = field(default_factory=dict)

    # --- SAD (System Architecture Document) ---
    sad_spec: dict = field(default_factory=dict)

    # --- FRD (Functional Requirements Document) ---
    frd_spec: dict = field(default_factory=dict)

    # --- Architecture decisions (populated by specialists) ---
    block_diagram: dict = field(default_factory=dict)
    interface_contracts: dict = field(default_factory=dict)
    memory_map: dict = field(default_factory=dict)
    clock_tree: dict = field(default_factory=dict)
    register_spec: dict = field(default_factory=dict)
    power_budget: dict = field(default_factory=dict)
    area_budget: dict = field(default_factory=dict)

    # --- Benchmark results ---
    benchmark_results: dict = field(default_factory=dict)

    # --- Questions ---
    pending_questions: list[dict] = field(default_factory=list)
    answered_questions: list[dict] = field(default_factory=list)

    # --- Output ---
    block_specs: list[dict] = field(default_factory=list)
    block_diagram_doc: dict = field(default_factory=dict)
    block_diagram_doc_path: str = ""

    def add_question(self, question: ArchitectureQuestion) -> None:
        """Add a question from a specialist agent."""
        self.pending_questions.append(question.to_dict())

    def answer_question(self, question_id: str, answer: str) -> bool:
        """Answer a pending question and move it to answered list.

        Returns True if the question was found and answered.
        """
        for i, q in enumerate(self.pending_questions):
            if q.get("id") == question_id:
                q["answer"] = answer
                q["answered_at"] = datetime.now(UTC).isoformat()
                self.answered_questions.append(q)
                self.pending_questions.pop(i)
                return True
        return False

    def has_blocking_questions(self) -> bool:
        """Check if there are any unanswered blocking questions."""
        return any(
            q.get("priority") == "blocking" and q.get("answer") is None
            for q in self.pending_questions
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureState:
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


# ---------------------------------------------------------------------------
# State file I/O
# ---------------------------------------------------------------------------


def _state_path(project_root: str) -> Path:
    return Path(project_root) / STATE_DIR / STATE_FILE


def load_state(project_root: str) -> ArchitectureState:
    """Load architecture state from disk.

    Returns a fresh ArchitectureState if the file doesn't exist.
    """
    path = _state_path(project_root)
    if not path.exists():
        return ArchitectureState()
    data = json.loads(path.read_text())
    return ArchitectureState.from_dict(data)


def save_state(state: ArchitectureState, project_root: str) -> None:
    """Write architecture state to disk as JSON (atomic write)."""
    from orchestrator.utils import atomic_write

    path = _state_path(project_root)
    atomic_write(path, json.dumps(state.to_dict(), indent=2, default=str))


def clear_state(project_root: str) -> None:
    """Remove the state file to start fresh."""
    path = _state_path(project_root)
    if path.exists():
        path.unlink()
