# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared test fixtures for the coresmith orchestrator test suite.

Provides:
- State isolation fixtures (isolated_project, fft16_initial_state) that use
  tmp_path so tests never touch the real .coresmith/ directory.
- Graph fixtures (arch_graph, pipeline_graph, backend_graph) with in-memory
  MemorySaver checkpointers -- no SQLite files created.
- MCP server reset fixture for tests that exercise the tool layer.
- Per-document state fixtures (fft16_full_docs) for the document hierarchy.
- Assertion helpers (assert_doc_files) for validating document file pairs.

FFT16 reference design constants live in fft16_fixtures.py (importable module).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# A-Fix 1: pin the LEGACY profile for the whole test session
# ═══════════════════════════════════════════════════════════════════════════
# Hundreds of existing assertions pin default-OFF gate behavior. The STRICT
# profile (the production default) would seed those gates ON. Hard-set legacy
# at conftest import -- before any module applies the profile -- so the suite's
# historical defaults hold. Tests that exercise strict (test_profile.py) use
# monkeypatch to switch temporarily; it reverts to this value afterwards.
os.environ["CORESMITH_PROFILE"] = "legacy"

from orchestrator.tests.fft16_fixtures import (
    FFT16_BLOCK_DIAGRAM,
    FFT16_CLOCK_TREE,
    FFT16_FRD_MARKDOWN,
    FFT16_MEMORY_MAP,
    FFT16_PRD_DOCUMENT,
    FFT16_REGISTER_SPEC,
    FFT16_REQUIREMENTS,
    FFT16_SAD_MARKDOWN,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fix import-time side effect in constraints module
# ═══════════════════════════════════════════════════════════════════════════
# orchestrator/architecture/constraints.py reads a prompt file at import time
# using a path that resolves incorrectly (parents[2] misses 'orchestrator/').
# Create the expected file so the import doesn't fail during tests.

def _ensure_constraint_prompt():
    """Create the constraint_check.md symlink/copy if missing."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "langchain" / "prompts" / "constraint_check.md"
    target_dir = Path(__file__).resolve().parents[2] / "langchain" / "prompts"
    target = target_dir / "constraint_check.md"
    if not target.exists() and src.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(src.read_text())

_ensure_constraint_prompt()


@pytest.fixture(autouse=True)
def _reset_profile_state():
    """Clear profile apply-state around every test.

    ``apply()`` is idempotent per-process; resetting before/after each test means
    a test that switches to strict (and thus seeds gate env vars) cannot leak
    those seeded vars into the next test. Legacy (the conftest pin) seeds
    nothing, so the common case is a no-op.
    """
    from orchestrator import profile as _profile
    _profile.reset()
    yield
    _profile.reset()


@pytest.fixture
def replay_llm(monkeypatch):
    """Route ``ClaudeLLM`` through the record/replay backend (Package C, C4).

    Yields a loader ``load(name, *, strict=True, project_root=None) -> ReplayBackend``
    that selects ``CORESMITH_LLM_PROVIDER=replay`` and loads the named fixture from
    ``orchestrator/tests/fixtures/replay/<name>``. The backend is a module
    singleton, so it is reset before load and after the test.
    """
    from orchestrator.testing import replay_provider as rp

    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "replay")
    monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
    fixtures_dir = Path(__file__).parent / "fixtures" / "replay"

    def _load(name, *, strict=True, project_root=None):
        rp.reset_backend()
        path = Path(name) if os.path.isabs(str(name)) else fixtures_dir / name
        backend = rp.set_fixture(path, strict=strict)
        if project_root is not None:
            monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(project_root))
        return backend

    yield _load
    rp.reset_backend()


@pytest.fixture
async def from_stage(tmp_path, monkeypatch):
    """Materialize a stage fixture into a throwaway project root (Package C, C4/C5).

    Yields an async loader ``load(name, *, project_root=None, strict=None) ->
    StageContext``. The AsyncSqliteSaver each StageContext opens is closed in
    teardown (pytest hangs at exit otherwise -- plan C7). Fingerprint drift skips
    (or raises under CORESMITH_STAGE_STRICT).
    """
    from orchestrator.testing import stage_fixtures as sf

    fixtures_dir = Path(__file__).parent / "fixtures" / "stage"
    created = []

    async def _load(name, *, project_root=None, strict=None):
        pr = Path(project_root) if project_root else (tmp_path / "stage_root")
        pr.mkdir(parents=True, exist_ok=True)
        path = Path(name) if os.path.isabs(str(name)) else fixtures_dir / name
        ctx = await sf.materialize_stage(path, str(pr), monkeypatch, strict=strict)
        created.append(ctx)
        return ctx

    yield _load
    for ctx in created:
        await ctx.aclose()


@pytest.fixture
def isolated_project(tmp_path):
    """Temporary project root with a clean .coresmith/ directory.

    Use for any test that writes state to disk (ERS, block_specs, etc.).
    The tmp_path is auto-deleted by pytest after the test.
    """
    coresmith_dir = tmp_path / ".coresmith"
    coresmith_dir.mkdir()
    return str(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Fixtures (in-memory checkpointer -- no SQLite)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def arch_graph():
    """Fresh architecture graph with in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from orchestrator.langgraph.architecture_graph import build_architecture_graph

    return build_architecture_graph(checkpointer=MemorySaver())


@pytest.fixture
def pipeline_graph():
    """Fresh pipeline graph with in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from orchestrator.langgraph.pipeline_graph import build_pipeline_graph

    return build_pipeline_graph(checkpointer=MemorySaver())


@pytest.fixture
def backend_graph():
    """Fresh backend graph with in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from orchestrator.langgraph.backend_graph import build_backend_graph

    return build_backend_graph(checkpointer=MemorySaver())


@pytest.fixture
def fft16_initial_state(isolated_project):
    """Initial ArchGraphState dict for the FFT16 reference design.

    Points project_root at an isolated tmp_path so all disk writes
    (PRD, block_specs, events) go to a fresh ephemeral directory.
    """
    pdk_summary = "sky130 | 130nm | 1.8V | tt_025C_1v80"
    return {
        "project_root": isolated_project,
        "requirements": FFT16_REQUIREMENTS,
        "pdk_summary": pdk_summary,
        "target_clock_mhz": 50.0,
        "pdk_config": {},
        "max_rounds": 3,
        "round": 1,
        "phase": "prd",
        "prd_spec": None,
        "prd_questions": None,
        "violations_history": [],
        "questions": [],
        "block_diagram": None,
        "memory_map": None,
        "clock_tree": None,
        "register_spec": None,
        "benchmark_data": None,
        "constraint_result": None,
        "human_feedback": "",
        "human_response": None,
        "success": False,
        "error": "",
        "block_specs_path": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-Document State Fixtures
# ═══════════════════════════════════════════════════════════════════════════

# JSON documents (stored as .json)
_FFT16_JSON_DOC_MAP = {
    "prd_spec.json": FFT16_PRD_DOCUMENT,
    "block_diagram.json": FFT16_BLOCK_DIAGRAM,
    "memory_map.json": FFT16_MEMORY_MAP,
    "clock_tree.json": FFT16_CLOCK_TREE,
    "register_spec.json": FFT16_REGISTER_SPEC,
}

# Markdown documents (stored as .md -- SAD and FRD)
_FFT16_MD_DOC_MAP = {
    "sad_spec.md": FFT16_SAD_MARKDOWN["sad_text"],
    "frd_spec.md": FFT16_FRD_MARKDOWN["frd_text"],
}


@pytest.fixture
def fft16_full_docs(isolated_project):
    """Isolated project with all per-document files pre-populated.

    JSON documents are written as .json, SAD/FRD as .md (markdown only).
    Use for tests that start mid-flow or need to verify consumer reads
    against the per-document state architecture.
    """
    coresmith = Path(isolated_project) / ".coresmith"
    for name, data in _FFT16_JSON_DOC_MAP.items():
        (coresmith / name).write_text(json.dumps(data, indent=2))
    arch = Path(isolated_project) / "arch"
    arch.mkdir(parents=True, exist_ok=True)
    for name, text in _FFT16_MD_DOC_MAP.items():
        (arch / name).write_text(text)
    return isolated_project


# ═══════════════════════════════════════════════════════════════════════════
# Architecture-graph LLM containment
# ═══════════════════════════════════════════════════════════════════════════

def enter_arch_llm_node_patches(stack) -> None:
    """Shut the architecture-graph nodes that reach a live LLM but are not
    "specialists", so a test driving the whole graph stays offline.

    Every test module that runs the architecture graph maintains its own list
    of specialist patches, and both lists independently missed these two: they
    are graph nodes rather than ``architecture.specialists.*`` modules, so they
    do not look like something a "patch all specialists" helper should cover.
    The result was that tests carrying no ``live_llm`` marker made real, billed
    API calls, and the agents' shell tools ran unbounded commands (one observed
    run spent 30+ minutes on a filesystem-wide ``find /``). The failure mode is
    load-bearing on latency, not correctness, so it hid in plain sight -- CI
    just looked slow.

    Kept here rather than duplicated per module so the containment cannot drift
    out of sync again. A test that genuinely wants either node's real behaviour
    should mark itself ``live_llm`` and not call this.

    Args:
        stack: An ``ExitStack`` the caller already owns; patches are entered
            into it and unwound with it.
    """
    from unittest.mock import AsyncMock, patch

    stack.enter_context(patch(
        "orchestrator.architecture.specialists.interface_definition"
        ".analyze_interface_definition",
        new_callable=AsyncMock,
        return_value={
            "result": {
                "design_summary": "patched by enter_arch_llm_node_patches",
                "contracts": [],
                "contract_violations": [],
                "open_questions": [],
            },
            "questions": [],
        },
    ))
    stack.enter_context(patch(
        "orchestrator.langchain.agents.output_contract_review_agent"
        ".OutputContractReviewAgent.review",
        new_callable=AsyncMock,
        return_value={
            "passed": True,
            "orphaned_properties": [],
            "summary": "patched by enter_arch_llm_node_patches",
            "feedback_for_redecomposition": "",
        },
    ))


# ═══════════════════════════════════════════════════════════════════════════
# Assertion Helpers
# ═══════════════════════════════════════════════════════════════════════════

_MD_ONLY_DOCS = {"sad_spec", "frd_spec"}


def assert_doc_files(project_root: str, expected: list[str]) -> None:
    """Assert that each doc has .md in arch/ and optionally .json in .coresmith/.

    SAD and FRD are markdown-only (no .json). All others have both.

    Args:
        project_root: Path to the project root.
        expected: List of document base names (e.g. ["prd_spec", "sad_spec"]).

    Raises AssertionError with a descriptive message on failure.
    """
    coresmith = Path(project_root) / ".coresmith"
    arch = Path(project_root) / "arch"
    for doc in expected:
        md_path = arch / f"{doc}.md"
        assert md_path.exists(), f"Missing {md_path}"
        md_text = md_path.read_text()
        assert md_text.startswith("#"), f"{md_path} doesn't start with a heading"

        if doc not in _MD_ONLY_DOCS:
            json_path = coresmith / f"{doc}.json"
            assert json_path.exists(), f"Missing {json_path}"
            data = json.loads(json_path.read_text())
            assert isinstance(data, dict), f"{json_path} is not a JSON object"


# ═══════════════════════════════════════════════════════════════════════════
# MCP Server Reset Fixture
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def reset_mcp_state(tmp_path, monkeypatch):
    """Reset MCP server module-level singletons and redirect all state
    to a fresh temporary directory.  Prevents SQLite contamination.

    Usage: request this fixture in any test_mcp_server.py test.
    """
    import orchestrator.mcp_server as mcp

    (tmp_path / ".coresmith").mkdir()

    monkeypatch.setattr(mcp, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        mcp, "_ARCH_CHECKPOINT_DB",
        str(tmp_path / ".coresmith" / "architecture_checkpoint.db"),
    )
    monkeypatch.setattr(
        mcp, "_CHECKPOINT_DB",
        str(tmp_path / ".coresmith" / "pipeline_checkpoint.db"),
    )
    monkeypatch.setattr(
        mcp, "_BACKEND_CHECKPOINT_DB",
        str(tmp_path / ".coresmith" / "backend_checkpoint.db"),
    )

    mcp._architecture = mcp.GraphLifecycle(
        name="architecture",
        checkpoint_db=str(tmp_path / ".coresmith" / "architecture_checkpoint.db"),
        builder_fn_path="orchestrator.langgraph.architecture_graph",
        builder_fn_name="build_architecture_graph",
    )
    mcp._pipeline = mcp.GraphLifecycle(
        name="pipeline",
        checkpoint_db=str(tmp_path / ".coresmith" / "pipeline_checkpoint.db"),
        builder_fn_path="orchestrator.langgraph.pipeline_graph",
        builder_fn_name="build_pipeline_graph",
    )
    mcp._backend = mcp.GraphLifecycle(
        name="backend",
        checkpoint_db=str(tmp_path / ".coresmith" / "backend_checkpoint.db"),
        builder_fn_path="orchestrator.langgraph.backend_graph",
        builder_fn_name="build_backend_graph",
    )

    yield


# ═══════════════════════════════════════════════════════════════════════════
# Async Polling Helper
# ═══════════════════════════════════════════════════════════════════════════

async def wait_for_status(runner, target_statuses: set[str], timeout: float = 15.0) -> str:
    """Poll a GraphLifecycle runner until its status is in *target_statuses*.

    Returns the matching status. Raises ``TimeoutError`` if the timeout
    expires before the status matches.

    Args:
        runner: A ``GraphLifecycle`` instance (e.g. ``mcp._architecture``).
        target_statuses: Set of status strings to wait for.
        timeout: Maximum seconds to wait (default 15).
    """
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runner.status in target_statuses:
            return runner.status
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"Runner '{runner.name}' status is '{runner.status}', "
        f"expected one of {target_statuses} within {timeout}s"
    )
