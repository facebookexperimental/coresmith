# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""_die_cap_excludes_sram: std-cell cap SRAM-exclusion detection across
PRD + ERS with literal markers + proximity match; env-gated legacy mode."""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.langgraph.pipeline_graph import _die_cap_excludes_sram


def _write(root: Path, fn: str, text: str) -> None:
    d = root / ".coresmith"
    d.mkdir(parents=True, exist_ok=True)
    (d / fn).write_text(json.dumps({"spec": text}))


def test_legacy_literal_marker_in_prd(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", raising=False)
    _write(tmp_path, "prd_spec.json", "SRAM black boxes excluded from the cap")
    assert _die_cap_excludes_sram(str(tmp_path)) is True


def test_live_run_phrasing_prd(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", raising=False)
    _write(tmp_path, "prd_spec.json",
           "macros are blackboxed out of the scored standard-cell count, "
           "flip-flop count, and placed standard-cell area.")
    assert _die_cap_excludes_sram(str(tmp_path)) is True


def test_live_run_phrasing_ers_only(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", raising=False)
    _write(tmp_path, "prd_spec.json", "nothing relevant here")
    _write(tmp_path, "ers_spec.json",
           "2,320,000 um2 placed standard-cell area, excluding the named "
           "SRAM macro instances exactly once.")
    assert _die_cap_excludes_sram(str(tmp_path)) is True


def test_no_markers_false(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", raising=False)
    _write(tmp_path, "prd_spec.json",
           "QSPI transaction latency is host-paced and excluded from "
           "compute throughput. die area budget 2.32 mm2.")
    assert _die_cap_excludes_sram(str(tmp_path)) is False


def test_env_restores_legacy_prd_only_literals(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", "0")
    # proximity-only phrasing no longer matches...
    _write(tmp_path, "prd_spec.json",
           "excluding the named SRAM macro instances exactly once")
    assert _die_cap_excludes_sram(str(tmp_path)) is False
    # ...ERS is not consulted...
    _write(tmp_path, "ers_spec.json", "memories blackboxed")
    assert _die_cap_excludes_sram(str(tmp_path)) is False
    # ...but a PRD literal still works.
    _write(tmp_path, "prd_spec.json", "memories blackboxed")
    assert _die_cap_excludes_sram(str(tmp_path)) is True


def test_missing_files_false(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", raising=False)
    assert _die_cap_excludes_sram(str(tmp_path)) is False


class TestContainerSubsume:
    def _item(self, name, area):
        from orchestrator.langgraph.mem_price import RollupItem
        return RollupItem(name=name, area_um2=area, source="ppa_history")

    def test_leaves_win_when_larger(self, monkeypatch):
        from orchestrator.langgraph.pipeline_graph import _subsume_container_items
        monkeypatch.delenv("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", raising=False)
        items = [self._item("wrapper", 250000.0),
                 self._item("a", 117000.0), self._item("b", 137000.0)]
        out, note = _subsume_container_items(items, {"wrapper"})
        assert [i.name for i in out] == ["a", "b"]
        assert "subsumed by their leaves" in note

    def test_container_wins_when_larger(self, monkeypatch):
        from orchestrator.langgraph.pipeline_graph import _subsume_container_items
        monkeypatch.delenv("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", raising=False)
        items = [self._item("wrapper", 300000.0),
                 self._item("a", 100000.0), self._item("b", 100000.0)]
        out, note = _subsume_container_items(items, {"wrapper"})
        assert [i.name for i in out] == ["wrapper"]
        assert "leaf blocks subsumed" in note

    def test_no_containers_unchanged(self, monkeypatch):
        from orchestrator.langgraph.pipeline_graph import _subsume_container_items
        monkeypatch.delenv("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", raising=False)
        items = [self._item("a", 1.0), self._item("b", 2.0)]
        out, note = _subsume_container_items(items, set())
        assert out == items and note == ""

    def test_env_restores_legacy_sum(self, monkeypatch):
        from orchestrator.langgraph.pipeline_graph import _subsume_container_items
        monkeypatch.setenv("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", "0")
        items = [self._item("wrapper", 250000.0), self._item("a", 117000.0)]
        out, note = _subsume_container_items(items, {"wrapper"})
        assert out == items and note == ""

    def test_container_detection_without_rtl_target(self, tmp_path):
        # older runs record rtl_sha1 but NOT rtl_target: glob rtl/**/<name>.v
        # and prefer the sha1 match
        import hashlib
        import json as _json

        from orchestrator.langgraph.pipeline_graph import _container_block_names
        (tmp_path / "rtl" / "integration").mkdir(parents=True)
        leaf = tmp_path / "rtl" / "aes_core.v"
        leaf.write_text("module aes_core; endmodule\n")
        # decoy with the same basename that does NOT reference blocks
        decoy = tmp_path / "rtl" / "integration" / "user_project_wrapper.v"
        decoy.write_text("module user_project_wrapper; endmodule\n")
        real = tmp_path / "rtl" / "user_project_wrapper.v"
        real.write_text("module user_project_wrapper;\n"
                        "  aes_core u_core ();\nendmodule\n")
        sha = hashlib.sha1(real.read_bytes()).hexdigest()
        for name, extra in (("user_project_wrapper", {"rtl_sha1": sha}),
                            ("aes_core", {})):
            d = tmp_path / ".coresmith" / "blocks" / name
            d.mkdir(parents=True)
            (d / "best_result.json").write_text(
                _json.dumps({"sim_passed": True, **extra}))
        got = _container_block_names(
            str(tmp_path), ["user_project_wrapper", "aes_core"])
        assert got == {"user_project_wrapper"}

    def test_container_detection_via_include(self, tmp_path):
        import json as _json

        from orchestrator.langgraph.pipeline_graph import _container_block_names
        # leaf block
        (tmp_path / "rtl").mkdir()
        leaf = tmp_path / "rtl" / "aes_core.v"
        leaf.write_text("module aes_core; endmodule\n")
        # wrapper whose rtl `include`s the leaf file and instantiates it
        wrap = tmp_path / "rtl" / "wrapper_pads.v"
        wrap.write_text(f'`include "{leaf}"\n'
                        "module wrapper_pads;\n  aes_core u_core ();\nendmodule\n")
        for name, rel in (("user_project_wrapper", "rtl/wrapper_pads.v"),
                          ("aes_core", "rtl/aes_core.v")):
            d = tmp_path / ".coresmith" / "blocks" / name
            d.mkdir(parents=True)
            (d / "best_result.json").write_text(
                _json.dumps({"rtl_target": rel, "sim_passed": True}))
        got = _container_block_names(
            str(tmp_path), ["user_project_wrapper", "aes_core"])
        assert got == {"user_project_wrapper"}
