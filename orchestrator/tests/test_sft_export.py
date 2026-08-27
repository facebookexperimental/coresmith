"""Tests for the first-class SFT dataset emission (CORESMITH_EMIT_SFT)."""

import json
from pathlib import Path

import pytest

from orchestrator.langgraph import sft_export


def _make_run(tmp_path: Path, *, chip_pass=True, verified=("blk_a",),
              unverified=("blk_b",)) -> Path:
    root = tmp_path / "run"
    (root / "arch" / "uarch_specs").mkdir(parents=True)
    (root / "tb" / "cocotb").mkdir(parents=True)
    (root / "rtl" / "grp").mkdir(parents=True)
    (root / "rtl" / "integration").mkdir(parents=True)
    cs = root / ".coresmith"
    for blk in list(verified) + list(unverified):
        bdir = cs / "blocks" / blk
        bdir.mkdir(parents=True)
        (root / "arch" / "uarch_specs" / f"{blk}.md").write_text(
            f"# {blk} spec\n\nDoes things.\n")
        (root / "rtl" / "grp" / f"{blk}.v").write_text(
            f"module {blk} (\n  input clk,\n  output [7:0] q\n);\n"
            "endmodule\n")
        (root / "tb" / "cocotb" / f"test_{blk}.py").write_text(
            "import cocotb\n")
        (bdir / "constraints.json").write_text(json.dumps(
            [{"rule": "mb_last is a level flag", "source": "x"}]))
        (bdir / "attempt_history.json").write_text(json.dumps([{}, {}]))
    for blk in verified:
        (cs / "blocks" / blk / "best_result.json").write_text("{}")
    (cs / "block_diagram.json").write_text(json.dumps({
        "blocks": [{"name": b, "description": "d"}
                   for b in list(verified) + list(unverified)],
        "connections": [{"from_block": "blk_a", "to_block": "blk_b",
                         "interface": "bus"}],
    }))
    (root / "rtl" / "integration" / "chip.v").write_text(
        "module chip_top (\n  input clk\n);\nendmodule\n")
    (cs / "chip_top_sources.f").write_text(
        str(root / "rtl" / "integration" / "chip.v") + "\n")
    (cs / "engine_sha.json").write_text(json.dumps({"sha": "abc123"}))
    (root / "final_report.json").write_text(json.dumps(
        {"signoff": {"status": "PASS" if chip_pass else "FAIL"}}))
    return root


class TestEmitSftDataset:
    def test_layout_and_labels_chip_verified(self, tmp_path):
        root = _make_run(tmp_path)
        manifest = sft_export.emit_sft_dataset(str(root))
        assert (root / "sft" / "README.md").exists()
        assert (root / "sft" / "manifest.json").exists()
        assert manifest["chip_verified"] is True
        assert manifest["skipped_unverified"] == ["blk_b"]
        assert manifest["engine_sha"] == "abc123"
        rows = [json.loads(ln) for ln in
                (root / "sft" / "pairs" / "uarch_to_rtl.jsonl")
                .read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["labels"]["verified"] == "chip"
        assert row["labels"]["block"] == "blk_a"
        assert row["labels"]["attempts"] == 2
        assert "MANDATORY BLOCK CONSTRAINTS" in row["messages"][1]["content"]
        assert "module blk_a" in row["messages"][2]["content"]

    def test_block_dv_tier_when_chip_failed(self, tmp_path):
        root = _make_run(tmp_path, chip_pass=False)
        sft_export.emit_sft_dataset(str(root))
        row = json.loads((root / "sft" / "pairs" / "uarch_to_rtl.jsonl")
                         .read_text().splitlines()[0])
        assert row["labels"]["verified"] == "block_dv"

    def test_all_four_pair_files(self, tmp_path):
        root = _make_run(tmp_path)
        manifest = sft_export.emit_sft_dataset(str(root))
        assert set(manifest["counts"]) == {
            "uarch_to_rtl", "uarch_to_testbench",
            "integration_to_chip_top", "spec_generation"}
        chip = json.loads(
            (root / "sft" / "pairs" / "integration_to_chip_top.jsonl")
            .read_text().splitlines()[0])
        assert "BLOCK DIAGRAM" in chip["messages"][1]["content"]
        assert "module blk_a" in chip["messages"][1]["content"]  # port contract
        assert "module chip_top" in chip["messages"][2]["content"]

    def test_no_verified_blocks_emits_chip_top_only_or_none(self, tmp_path):
        root = _make_run(tmp_path, verified=(), unverified=("blk_a", "blk_b"))
        manifest = sft_export.emit_sft_dataset(str(root))
        # Block pairs must all be absent; only the assembled chip-top pair
        # may remain.
        if manifest is not None:
            assert set(manifest["counts"]) <= {"integration_to_chip_top"}
        assert not (root / "sft" / "pairs" / "uarch_to_rtl.jsonl").exists()

    def test_empty_run_returns_none(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        assert sft_export.emit_sft_dataset(str(root)) is None
        assert not (root / "sft").exists()


class TestSftEnabledGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_EMIT_SFT", raising=False)
        assert sft_export.sft_enabled() is False

    def test_env_one_enables(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_EMIT_SFT", "1")
        assert sft_export.sft_enabled() is True
