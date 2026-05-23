# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the per-document file I/O layer (doc_store.py).

Tier 1: Pure unit tests -- no graph, no async.

Tests:
- Individual reader functions (read_prd, read_sad, read_frd, etc.)
- list_documents accuracy
- Round-trip: write fixture data -> read back -> assert equality
- Malformed JSON handling
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coresmith(tmp_path: Path) -> Path:
    """Create and return the .coresmith/ directory inside tmp_path."""
    d = tmp_path / ".coresmith"
    d.mkdir(exist_ok=True)
    return d


def _arch(tmp_path: Path) -> Path:
    """Create and return the arch/ directory inside tmp_path."""
    d = tmp_path / "arch"
    d.mkdir(exist_ok=True)
    return d


def _write_doc(coresmith_dir: Path, filename: str, data: dict) -> None:
    """Write a JSON document to the .coresmith/ directory."""
    (coresmith_dir / filename).write_text(json.dumps(data, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# Reader Functions
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestReadFunctions:
    """Test each doc_store reader returns the correct data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.root = str(tmp_path)
        self.coresmith = _coresmith(tmp_path)
        self.arch = _arch(tmp_path)

    def test_read_prd_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_prd
        assert read_prd(self.root) is None

    def test_read_prd_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_prd
        _write_doc(self.coresmith, "prd_spec.json", FFT16_PRD_DOCUMENT)
        result = read_prd(self.root)
        assert result is not None
        assert result["phase"] == "prd_complete"

    def test_read_sad_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_sad
        assert read_sad(self.root) is None

    def test_read_sad_returns_markdown_when_present(self):
        from orchestrator.architecture.doc_store import read_sad
        (self.arch / "sad_spec.md").write_text(FFT16_SAD_MARKDOWN["sad_text"])
        result = read_sad(self.root)
        assert result is not None
        assert "16-Point FFT Processor" in result

    def test_read_frd_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_frd
        assert read_frd(self.root) is None

    def test_read_frd_returns_markdown_when_present(self):
        from orchestrator.architecture.doc_store import read_frd
        (self.arch / "frd_spec.md").write_text(FFT16_FRD_MARKDOWN["frd_text"])
        result = read_frd(self.root)
        assert result is not None
        assert "16-Point FFT Processor" in result

    def test_read_block_diagram_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_block_diagram
        assert read_block_diagram(self.root) is None

    def test_read_block_diagram_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_block_diagram
        _write_doc(self.coresmith, "block_diagram.json", FFT16_BLOCK_DIAGRAM)
        result = read_block_diagram(self.root)
        assert result is not None
        assert len(result["blocks"]) == 3

    def test_read_memory_map_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_memory_map
        assert read_memory_map(self.root) is None

    def test_read_memory_map_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_memory_map
        _write_doc(self.coresmith, "memory_map.json", FFT16_MEMORY_MAP)
        result = read_memory_map(self.root)
        assert result is not None
        assert len(result["result"]["peripherals"]) == 3

    def test_read_clock_tree_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_clock_tree
        assert read_clock_tree(self.root) is None

    def test_read_clock_tree_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_clock_tree
        _write_doc(self.coresmith, "clock_tree.json", FFT16_CLOCK_TREE)
        result = read_clock_tree(self.root)
        assert result is not None
        assert result["result"]["num_domains"] == 1

    def test_read_register_spec_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_register_spec
        assert read_register_spec(self.root) is None

    def test_read_register_spec_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_register_spec
        _write_doc(self.coresmith, "register_spec.json", FFT16_REGISTER_SPEC)
        result = read_register_spec(self.root)
        assert result is not None
        assert result["result"]["total_blocks"] == 4

    def test_read_ers_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_ers
        assert read_ers(self.root) is None

    def test_read_ers_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_ers
        ers_data = {"ers": {"title": "Final ERS"}, "phase": "ers_complete"}
        _write_doc(self.coresmith, "ers_spec.json", ers_data)
        result = read_ers(self.root)
        assert result is not None
        assert result["ers"]["title"] == "Final ERS"

    def test_read_block_specs_returns_none_when_missing(self):
        from orchestrator.architecture.doc_store import read_block_specs
        assert read_block_specs(self.root) is None

    def test_read_block_specs_returns_data_when_present(self):
        from orchestrator.architecture.doc_store import read_block_specs
        specs = [{"name": "fft_butterfly", "tier": 1}]
        (self.coresmith / "block_specs.json").write_text(json.dumps(specs))
        result = read_block_specs(self.root)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "fft_butterfly"


# ═══════════════════════════════════════════════════════════════════════════
# Malformed JSON Handling
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestMalformedJson:
    """Verify readers handle corrupt/invalid JSON gracefully."""

    def test_malformed_json_returns_none(self, tmp_path):
        from orchestrator.architecture.doc_store import read_prd

        coresmith = _coresmith(tmp_path)
        (coresmith / "prd_spec.json").write_text("{invalid json")
        assert read_prd(str(tmp_path)) is None

    def test_empty_sad_file_returns_none(self, tmp_path):
        from orchestrator.architecture.doc_store import read_sad

        arch = _arch(tmp_path)
        (arch / "sad_spec.md").write_text("")
        assert read_sad(str(tmp_path)) is None

    def test_empty_frd_file_returns_none(self, tmp_path):
        from orchestrator.architecture.doc_store import read_frd

        arch = _arch(tmp_path)
        (arch / "frd_spec.md").write_text("")
        assert read_frd(str(tmp_path)) is None


# ═══════════════════════════════════════════════════════════════════════════
# list_documents
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestListDocuments:
    """Test list_documents() accuracy."""

    def test_all_false_on_empty_coresmith(self, tmp_path):
        from orchestrator.architecture.doc_store import list_documents

        _coresmith(tmp_path)
        result = list_documents(str(tmp_path))
        assert all(v is False for v in result.values())

    def test_reflects_present_documents(self, tmp_path):
        from orchestrator.architecture.doc_store import list_documents

        coresmith = _coresmith(tmp_path)
        arch = _arch(tmp_path)
        _write_doc(coresmith, "prd_spec.json", FFT16_PRD_DOCUMENT)
        _write_doc(coresmith, "block_diagram.json", FFT16_BLOCK_DIAGRAM)
        (arch / "sad_spec.md").write_text(FFT16_SAD_MARKDOWN["sad_text"])

        result = list_documents(str(tmp_path))
        assert result["prd_spec"] is True
        assert result["block_diagram"] is True
        assert result["sad_spec"] is True
        assert result["frd_spec"] is False

    def test_ignores_non_document_files(self, tmp_path):
        from orchestrator.architecture.doc_store import list_documents

        coresmith = _coresmith(tmp_path)
        (coresmith / "pipeline_events.jsonl").write_text("")
        (coresmith / "pipeline_checkpoint.db").write_text("")

        result = list_documents(str(tmp_path))
        assert all(v is False for v in result.values())

    def test_all_documents_present(self, tmp_path):
        from orchestrator.architecture.doc_store import list_documents

        coresmith = _coresmith(tmp_path)
        arch = _arch(tmp_path)
        json_docs = [
            "prd_spec", "block_diagram",
            "memory_map", "clock_tree", "register_spec", "ers_spec",
        ]
        for doc in json_docs:
            _write_doc(coresmith, f"{doc}.json", {"test": True})
        # SAD and FRD are now markdown-only (in arch/ directory)
        (arch / "sad_spec.md").write_text("# SAD")
        (arch / "frd_spec.md").write_text("# FRD")

        all_docs = json_docs + ["sad_spec", "frd_spec"]
        result = list_documents(str(tmp_path))
        assert all(result[d] is True for d in all_docs)


# ═══════════════════════════════════════════════════════════════════════════
# Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestRoundTrip:
    """Write FFT16 fixture data -> read via doc_store -> assert equality."""

    _JSON_CASES = [
        ("prd_spec.json", FFT16_PRD_DOCUMENT, "read_prd"),
        ("block_diagram.json", FFT16_BLOCK_DIAGRAM, "read_block_diagram"),
        ("memory_map.json", FFT16_MEMORY_MAP, "read_memory_map"),
        ("clock_tree.json", FFT16_CLOCK_TREE, "read_clock_tree"),
        ("register_spec.json", FFT16_REGISTER_SPEC, "read_register_spec"),
    ]

    @pytest.mark.parametrize("filename,fixture_data,reader_name", _JSON_CASES)
    def test_round_trip_json(self, tmp_path, filename, fixture_data, reader_name):
        import orchestrator.architecture.doc_store as ds

        coresmith = _coresmith(tmp_path)
        _write_doc(coresmith, filename, fixture_data)

        reader = getattr(ds, reader_name)
        result = reader(str(tmp_path))
        assert result == fixture_data

    def test_round_trip_sad_markdown(self, tmp_path):
        import orchestrator.architecture.doc_store as ds

        _arch(tmp_path)
        (tmp_path / "arch" / "sad_spec.md").write_text(FFT16_SAD_MARKDOWN["sad_text"])
        result = ds.read_sad(str(tmp_path))
        assert result == FFT16_SAD_MARKDOWN["sad_text"]

    def test_round_trip_frd_markdown(self, tmp_path):
        import orchestrator.architecture.doc_store as ds

        _arch(tmp_path)
        (tmp_path / "arch" / "frd_spec.md").write_text(FFT16_FRD_MARKDOWN["frd_text"])
        result = ds.read_frd(str(tmp_path))
        assert result == FFT16_FRD_MARKDOWN["frd_text"]
