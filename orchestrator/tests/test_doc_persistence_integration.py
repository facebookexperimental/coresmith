# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for per-document persistence and doc_store.

Tier 3: Real filesystem via tmp_path -- tests the full round-trip of
persist helpers writing files that doc_store readers then consume.

Tests:
- Doc store integration (list_documents accuracy with real files)
- Markdown format integrity (headings, section structure)
- JSON/Markdown consistency (titles match, counts match)
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
    FFT16_REGISTER_SPEC,
    FFT16_SAD_MARKDOWN,
)

# ═══════════════════════════════════════════════════════════════════════════
# Doc Store Integration (uses fft16_full_docs fixture)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestDocStoreIntegration:
    """Integration: doc_store readers work on real pre-populated files."""

    def test_list_documents_all_present(self, fft16_full_docs):
        from orchestrator.architecture.doc_store import list_documents

        result = list_documents(fft16_full_docs)
        expected_true = [
            "prd_spec", "sad_spec", "frd_spec", "block_diagram",
            "memory_map", "clock_tree", "register_spec",
        ]
        for doc in expected_true:
            assert result[doc] is True, f"Expected {doc} to be present"
        assert result["ers_spec"] is False

    def test_each_reader_returns_correct_data(self, fft16_full_docs):
        from orchestrator.architecture.doc_store import (
            read_block_diagram,
            read_clock_tree,
            read_frd,
            read_memory_map,
            read_prd,
            read_register_spec,
            read_sad,
        )

        assert read_prd(fft16_full_docs) == FFT16_PRD_DOCUMENT
        # SAD and FRD are now read as markdown text
        sad_result = read_sad(fft16_full_docs)
        assert sad_result is not None
        assert "16-Point FFT Processor" in sad_result
        frd_result = read_frd(fft16_full_docs)
        assert frd_result is not None
        assert "16-Point FFT Processor" in frd_result
        assert read_block_diagram(fft16_full_docs) == FFT16_BLOCK_DIAGRAM
        assert read_memory_map(fft16_full_docs) == FFT16_MEMORY_MAP
        assert read_clock_tree(fft16_full_docs) == FFT16_CLOCK_TREE
        assert read_register_spec(fft16_full_docs) == FFT16_REGISTER_SPEC

    def test_deleting_file_reflects_in_list_and_reader(self, fft16_full_docs):
        from orchestrator.architecture.doc_store import list_documents, read_sad

        arch = Path(fft16_full_docs) / "arch"
        (arch / "sad_spec.md").unlink()

        result = list_documents(fft16_full_docs)
        assert result["sad_spec"] is False
        assert read_sad(fft16_full_docs) is None

    def test_no_coresmith_dir_returns_all_none(self, tmp_path):
        from orchestrator.architecture.doc_store import (
            list_documents,
            read_prd,
            read_sad,
        )

        result = list_documents(str(tmp_path))
        assert all(v is False for v in result.values())
        assert read_prd(str(tmp_path)) is None
        assert read_sad(str(tmp_path)) is None


# ═══════════════════════════════════════════════════════════════════════════
# Markdown Format Integrity
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestMarkdownFormatIntegrity:
    """Verify each persist helper produces well-formed Markdown."""

    _PERSIST_CASES = [
        ("_persist_sad", "sad_spec.md", FFT16_SAD_MARKDOWN,
         ["## System Overview", "## Architecture Decisions", "## Risk Assessment"]),
        ("_persist_frd", "frd_spec.md", FFT16_FRD_MARKDOWN,
         ["## Performance Requirements", "## Interface Requirements", "## Resource Budgets"]),
        ("_persist_block_diagram", "block_diagram.md", FFT16_BLOCK_DIAGRAM,
         ["## Blocks", "## Connections"]),
        ("_persist_memory_map", "memory_map.md", FFT16_MEMORY_MAP,
         ["fft_butterfly", "0x10000000"]),
        ("_persist_clock_tree", "clock_tree.md", FFT16_CLOCK_TREE,
         ["clk_sys"]),
        ("_persist_register_spec", "register_spec.md", FFT16_REGISTER_SPEC,
         ["fft_butterfly", "top_csr"]),
    ]

    @pytest.mark.parametrize(
        "helper_name,md_filename,fixture_data,expected_content",
        _PERSIST_CASES,
    )
    def test_md_contains_expected_content(
        self, isolated_project, helper_name, md_filename, fixture_data, expected_content,
    ):
        import orchestrator.langgraph.architecture_graph as ag

        persist_fn = getattr(ag, helper_name)
        persist_fn(isolated_project, fixture_data)

        md_path = Path(isolated_project) / "arch" / md_filename
        assert md_path.exists(), f"Missing {md_path}"
        md_text = md_path.read_text()

        assert md_text.startswith("#"), f"{md_filename} doesn't start with a heading"

        for content in expected_content:
            assert content in md_text, (
                f"Expected '{content}' in {md_filename}, got:\n{md_text[:500]}"
            )

    @pytest.mark.parametrize(
        "helper_name,md_filename,fixture_data,expected_content",
        _PERSIST_CASES,
    )
    def test_md_no_none_values(
        self, isolated_project, helper_name, md_filename, fixture_data, expected_content,
    ):
        """Markdown should not contain literal 'None' for populated fields."""
        import orchestrator.langgraph.architecture_graph as ag

        persist_fn = getattr(ag, helper_name)
        persist_fn(isolated_project, fixture_data)

        md_path = Path(isolated_project) / "arch" / md_filename
        md_text = md_path.read_text()
        lines_with_none = [
            line for line in md_text.split("\n")
            if "None" in line and not line.strip().startswith("#")
        ]
        assert len(lines_with_none) == 0, (
            f"Found 'None' in {md_filename}:\n" + "\n".join(lines_with_none)
        )


# ═══════════════════════════════════════════════════════════════════════════
# JSON / Markdown Consistency
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestJsonMarkdownConsistency:
    """Verify that JSON and Markdown files contain consistent data."""

    def test_sad_persisted_as_markdown_only(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_sad

        _persist_sad(isolated_project, FFT16_SAD_MARKDOWN)

        arch = Path(isolated_project) / "arch"
        coresmith = Path(isolated_project) / ".coresmith"
        assert (arch / "sad_spec.md").exists()
        assert not (coresmith / "sad_spec.json").exists(), "SAD should no longer produce .json"
        md_text = (arch / "sad_spec.md").read_text()
        assert "16-Point FFT Processor" in md_text

    def test_frd_persisted_as_markdown_only(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_frd

        _persist_frd(isolated_project, FFT16_FRD_MARKDOWN)

        arch = Path(isolated_project) / "arch"
        coresmith = Path(isolated_project) / ".coresmith"
        assert (arch / "frd_spec.md").exists()
        assert not (coresmith / "frd_spec.json").exists(), "FRD should no longer produce .json"
        md_text = (arch / "frd_spec.md").read_text()
        assert "16-Point FFT Processor" in md_text

    def test_block_diagram_block_count_matches(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_block_diagram

        _persist_block_diagram(isolated_project, FFT16_BLOCK_DIAGRAM)

        coresmith = Path(isolated_project) / ".coresmith"
        arch = Path(isolated_project) / "arch"
        data = json.loads((coresmith / "block_diagram.json").read_text())
        md_text = (arch / "block_diagram.md").read_text()

        for block in data["blocks"]:
            assert block["name"] in md_text, (
                f"Block '{block['name']}' from JSON not found in Markdown"
            )

    def test_memory_map_peripherals_in_md(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_memory_map

        _persist_memory_map(isolated_project, FFT16_MEMORY_MAP)

        coresmith = Path(isolated_project) / ".coresmith"
        arch = Path(isolated_project) / "arch"
        data = json.loads((coresmith / "memory_map.json").read_text())
        md_text = (arch / "memory_map.md").read_text()

        for periph in data["result"]["peripherals"]:
            assert periph["name"] in md_text

    def test_register_spec_blocks_in_md(self, isolated_project):
        from orchestrator.langgraph.architecture_graph import _persist_register_spec

        _persist_register_spec(isolated_project, FFT16_REGISTER_SPEC)

        coresmith = Path(isolated_project) / ".coresmith"
        arch = Path(isolated_project) / "arch"
        data = json.loads((coresmith / "register_spec.json").read_text())
        md_text = (arch / "register_spec.md").read_text()

        for block in data["result"]["blocks"]:
            assert block["name"] in md_text
