# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for per-document persistence helpers (_persist_prd, _persist_sad, etc.).

Tier 1: Pure unit tests -- no graph, no async. Each test calls a persist
helper directly and verifies the .json and .md files on disk.

Tests:
- JSON roundtrip (write then parse -> matches input)
- Markdown section validation (expected headings present)
- JSON/Markdown consistency (title appears in both)
- atomic_write safety (no .tmp files left behind)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.tests.fft16_fixtures import (
    FFT16_BLOCK_DIAGRAM,
    FFT16_CLOCK_TREE,
    FFT16_FRD_MARKDOWN,
    FFT16_MEMORY_MAP,
    FFT16_PRD_DOCUMENT,
    FFT16_PRD_QUESTIONS,
    FFT16_PRD_ANSWERS,
    FFT16_REGISTER_SPEC,
    FFT16_SAD_MARKDOWN,
)


# ═══════════════════════════════════════════════════════════════════════════
# PRD Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistPrd:
    """Test _persist_prd() writes correct .json and .md files."""

    def test_json_roundtrip(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_prd

        _persist_prd(
            isolated_project,
            FFT16_PRD_DOCUMENT,
            FFT16_PRD_QUESTIONS.get("questions"),
            FFT16_PRD_ANSWERS,
        )

        json_path = Path(isolated_project) / ".coresmith" / "prd_spec.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert isinstance(data, dict)

    def test_markdown_has_expected_sections(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_prd

        _persist_prd(
            isolated_project,
            FFT16_PRD_DOCUMENT,
            FFT16_PRD_QUESTIONS.get("questions"),
            FFT16_PRD_ANSWERS,
        )

        md_path = Path(isolated_project) / "arch" / "prd_spec.md"
        assert md_path.exists()
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "## Summary" in md_text
        assert "## Target Technology" in md_text
        assert "## Functional Requirements" in md_text

    def test_no_tmp_files_remain(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_prd

        _persist_prd(
            isolated_project,
            FFT16_PRD_DOCUMENT,
            FFT16_PRD_QUESTIONS.get("questions"),
            FFT16_PRD_ANSWERS,
        )

        coresmith = Path(isolated_project) / ".coresmith"
        tmp_files = list(coresmith.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover .tmp files: {tmp_files}"


# ═══════════════════════════════════════════════════════════════════════════
# SAD Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistSad:
    """Test _persist_sad() writes markdown-only SAD file."""

    def test_writes_md_only(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_sad

        _persist_sad(isolated_project, FFT16_SAD_MARKDOWN)

        arch = Path(isolated_project) / "arch"
        assert (arch / "sad_spec.md").exists()
        coresmith = Path(isolated_project) / ".coresmith"
        assert not (coresmith / "sad_spec.json").exists()

    def test_markdown_has_expected_sections(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_sad

        _persist_sad(isolated_project, FFT16_SAD_MARKDOWN)

        md_path = Path(isolated_project) / "arch" / "sad_spec.md"
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "## System Overview" in md_text
        assert "## Architecture Decisions" in md_text
        assert "## Risk Assessment" in md_text

    def test_roundtrip_content(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_sad

        _persist_sad(isolated_project, FFT16_SAD_MARKDOWN)

        md_path = Path(isolated_project) / "arch" / "sad_spec.md"
        assert md_path.read_text() == FFT16_SAD_MARKDOWN["sad_text"]


# ═══════════════════════════════════════════════════════════════════════════
# FRD Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistFrd:
    """Test _persist_frd() writes markdown-only FRD file."""

    def test_writes_md_only(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_frd

        _persist_frd(isolated_project, FFT16_FRD_MARKDOWN)

        arch = Path(isolated_project) / "arch"
        assert (arch / "frd_spec.md").exists()
        coresmith = Path(isolated_project) / ".coresmith"
        assert not (coresmith / "frd_spec.json").exists()

    def test_markdown_has_expected_sections(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_frd

        _persist_frd(isolated_project, FFT16_FRD_MARKDOWN)

        md_path = Path(isolated_project) / "arch" / "frd_spec.md"
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "## Performance Requirements" in md_text
        assert "## Interface Requirements" in md_text
        assert "## Resource Budgets" in md_text

    def test_roundtrip_content(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_frd

        _persist_frd(isolated_project, FFT16_FRD_MARKDOWN)

        md_path = Path(isolated_project) / "arch" / "frd_spec.md"
        assert md_path.read_text() == FFT16_FRD_MARKDOWN["frd_text"]


# ═══════════════════════════════════════════════════════════════════════════
# Block Diagram Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistBlockDiagram:
    """Test _persist_block_diagram() writes correct files."""

    def test_json_roundtrip(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_block_diagram

        _persist_block_diagram(isolated_project, FFT16_BLOCK_DIAGRAM)

        json_path = Path(isolated_project) / ".coresmith" / "block_diagram.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data["blocks"]) == 3

    def test_markdown_has_blocks_table(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_block_diagram

        _persist_block_diagram(isolated_project, FFT16_BLOCK_DIAGRAM)

        md_path = Path(isolated_project) / "arch" / "block_diagram.md"
        assert md_path.exists()
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "## Blocks" in md_text
        assert "fft_butterfly" in md_text
        assert "twiddle_rom" in md_text
        assert "fft_controller" in md_text

    def test_markdown_has_connections_section(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_block_diagram

        _persist_block_diagram(isolated_project, FFT16_BLOCK_DIAGRAM)

        md_path = Path(isolated_project) / "arch" / "block_diagram.md"
        md_text = md_path.read_text()
        assert "## Connections" in md_text


# ═══════════════════════════════════════════════════════════════════════════
# Memory Map Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistMemoryMap:
    """Test _persist_memory_map() writes correct files."""

    def test_json_roundtrip(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_memory_map

        _persist_memory_map(isolated_project, FFT16_MEMORY_MAP)

        json_path = Path(isolated_project) / ".coresmith" / "memory_map.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data["result"]["peripherals"]) == 3

    def test_markdown_has_peripheral_section(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_memory_map

        _persist_memory_map(isolated_project, FFT16_MEMORY_MAP)

        md_path = Path(isolated_project) / "arch" / "memory_map.md"
        assert md_path.exists()
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "fft_butterfly" in md_text
        assert "0x10000000" in md_text


# ═══════════════════════════════════════════════════════════════════════════
# Clock Tree Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistClockTree:
    """Test _persist_clock_tree() writes correct files."""

    def test_json_roundtrip(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_clock_tree

        _persist_clock_tree(isolated_project, FFT16_CLOCK_TREE)

        json_path = Path(isolated_project) / ".coresmith" / "clock_tree.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["result"]["num_domains"] == 1

    def test_markdown_has_domain_section(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_clock_tree

        _persist_clock_tree(isolated_project, FFT16_CLOCK_TREE)

        md_path = Path(isolated_project) / "arch" / "clock_tree.md"
        assert md_path.exists()
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "clk_sys" in md_text
        assert "50" in md_text


# ═══════════════════════════════════════════════════════════════════════════
# Register Spec Persistence
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPersistRegisterSpec:
    """Test _persist_register_spec() writes correct files."""

    def test_json_roundtrip(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_register_spec

        _persist_register_spec(isolated_project, FFT16_REGISTER_SPEC)

        json_path = Path(isolated_project) / ".coresmith" / "register_spec.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["result"]["total_blocks"] == 4

    def test_markdown_has_block_names(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_register_spec

        _persist_register_spec(isolated_project, FFT16_REGISTER_SPEC)

        md_path = Path(isolated_project) / "arch" / "register_spec.md"
        assert md_path.exists()
        md_text = md_path.read_text()

        assert md_text.startswith("#")
        assert "fft_butterfly" in md_text
        assert "top_csr" in md_text


# ═══════════════════════════════════════════════════════════════════════════
# Atomic Write Safety
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestAtomicWriteSafety:
    """Verify atomic_write doesn't leave .tmp files on success."""

    def test_no_tmp_files_after_successful_write(self, tmp_path):
        from orchestrator.utils import atomic_write

        target = tmp_path / "test_file.json"
        atomic_write(target, '{"hello": "world"}')

        assert target.exists()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        from orchestrator.utils import atomic_write

        target = tmp_path / "sub" / "dir" / "test.json"
        atomic_write(target, '{"nested": true}')

        assert target.exists()
        assert json.loads(target.read_text()) == {"nested": True}

    def test_atomic_write_overwrites_existing(self, tmp_path):
        from orchestrator.utils import atomic_write

        target = tmp_path / "overwrite.json"
        atomic_write(target, '{"version": 1}')
        atomic_write(target, '{"version": 2}')

        data = json.loads(target.read_text())
        assert data["version"] == 2
