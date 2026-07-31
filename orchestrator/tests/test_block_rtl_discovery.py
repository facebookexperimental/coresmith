# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A block whose Verilog file is not named after the block must still be found.

`discover_block_rtl` used to resolve a block's RTL only from the block result's
own ``rtl_path`` or by assuming ``<block_name>.v``. It never read ``rtl_target``,
the engine-controlled spec field that carries the real path. Every ordinary
block survives that, because its file IS named after it -- but a block whose
module name is LOCKED by an interface contract (a Caravel
``user_project_wrapper``, a vendor-mandated top) is exactly the block whose file
is named something else, and it was dropped from the returned dict with no
error at all.

Measured on run exp-raster-macro-20260727, where the spec said
``user_project_wrapper_io -> rtl/user_project_wrapper.v``:

    7 of 8 blocks parsed, pad adapter absent from `modules`
      -> detect_wrapper_block() returned None
      -> the DEFAULT-ON deterministic Caravel assembler never ran
      -> the LLM Integration Lead promoted the pad block's own ports to the chip
         boundary instead of instantiating the block
      -> the assembled top, the flat netlist and the GDS carried NO
         io_in / io_out / io_oeb -- the graded pinout did not exist
      -> the bus classifier found no Caravel boundary, integration DV fell back
         to the co-tuned LLM BFM, and the run reported PASS.

One silent omission, the full length of the flow. These tests cover both halves
of the fix: resolve ``rtl_target``, and make an unresolvable block impossible to
lose quietly.
"""
from __future__ import annotations

from orchestrator.langgraph.integration_helpers import (
    detect_wrapper_block,
    discover_block_rtl,
    parse_verilog_ports,
    unresolved_block_rtl,
)

PAD_ADAPTER = """
module user_project_wrapper (
    input  wire [37:0] io_in,
    output wire [37:0] io_out,
    output wire [37:0] io_oeb,
    input  wire        wb_clk_i,
    input  wire        wb_rst_i
);
endmodule
"""


def _leaf(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"module {name} (input wire clk);\nendmodule\n")
    return str(path)


class TestRtlTargetIsHonored:
    def test_file_not_named_after_the_block_is_found(self, tmp_path):
        """THE regression. A locked module name means block name != file name."""
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "user_project_wrapper.v").write_text(PAD_ADAPTER)
        found = discover_block_rtl(str(tmp_path), [{
            "name": "user_project_wrapper_io",
            "rtl_target": "rtl/user_project_wrapper.v",
        }])
        assert found == {
            "user_project_wrapper_io":
                str(tmp_path / "rtl" / "user_project_wrapper.v"),
        }

    def test_absolute_rtl_target_resolves(self, tmp_path):
        p = _leaf(tmp_path / "rtl" / "odd" / "weird_name.v", "blk")
        found = discover_block_rtl(str(tmp_path), [
            {"name": "blk", "rtl_target": p},
        ])
        assert found["blk"] == p

    def test_block_result_rtl_path_still_wins(self, tmp_path):
        """State from the completed block is more specific than the spec."""
        actual = _leaf(tmp_path / "rtl" / "from_state.v", "blk")
        _leaf(tmp_path / "rtl" / "from_spec.v", "blk")
        found = discover_block_rtl(str(tmp_path), [{
            "name": "blk",
            "rtl_path": actual,
            "rtl_target": "rtl/from_spec.v",
        }])
        assert found["blk"] == actual

    def test_stale_rtl_target_falls_through_to_convention(self, tmp_path):
        """A spec pointing at a file that is not there must not shadow a file
        that IS there -- otherwise a stale spec deletes a real block."""
        conventional = _leaf(tmp_path / "rtl" / "blk.v", "blk")
        found = discover_block_rtl(str(tmp_path), [{
            "name": "blk",
            "rtl_target": "rtl/moved_away.v",
        }])
        assert found["blk"] == conventional


class TestConventionalDiscoveryUnchanged:
    def test_rtl_subdir_layout(self, tmp_path):
        p = _leaf(tmp_path / "rtl" / "io_subsystem" / "front.v", "front")
        assert discover_block_rtl(str(tmp_path), [{"name": "front"}]) == {
            "front": p}

    def test_rtl_flat_layout(self, tmp_path):
        p = _leaf(tmp_path / "rtl" / "front.v", "front")
        assert discover_block_rtl(str(tmp_path), [{"name": "front"}]) == {
            "front": p}

    def test_per_block_dir_layout(self, tmp_path):
        p = _leaf(tmp_path / "rtl" / "front" / "front.v", "front")
        assert discover_block_rtl(str(tmp_path), [{"name": "front"}]) == {
            "front": p}


class TestAMissingBlockIsNotLostQuietly:
    def test_unresolved_block_is_reported(self, tmp_path):
        _leaf(tmp_path / "rtl" / "present.v", "present")
        missing = unresolved_block_rtl(str(tmp_path), [
            {"name": "present"},
            {"name": "vanished", "rtl_target": "rtl/nowhere.v"},
        ])
        assert missing == ["vanished"]

    def test_nothing_unresolved_when_all_are_found(self, tmp_path):
        _leaf(tmp_path / "rtl" / "a.v", "a")
        (tmp_path / "rtl" / "pad.v").write_text(PAD_ADAPTER)
        assert unresolved_block_rtl(str(tmp_path), [
            {"name": "a"},
            {"name": "pad_io", "rtl_target": "rtl/pad.v"},
        ]) == []

    def test_skipped_and_aborted_blocks_are_not_unresolved(self, tmp_path):
        """They are deliberately not in the chip, so they are not missing."""
        assert unresolved_block_rtl(str(tmp_path), [
            {"name": "gone", "skipped": True},
            {"name": "dead", "aborted": True},
        ]) == []


class TestTheConsequenceThatActuallyBit:
    def test_pad_adapter_is_detected_once_it_is_discovered(self, tmp_path):
        """The whole point. detect_wrapper_block can only see blocks that
        discovery returned, so dropping the pad block silently disabled the
        deterministic Caravel assembly and produced an ungradeable chip."""
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "user_project_wrapper.v").write_text(PAD_ADAPTER)
        _leaf(tmp_path / "rtl" / "core" / "engine.v", "engine")

        blocks = [
            {"name": "user_project_wrapper_io",
             "rtl_target": "rtl/user_project_wrapper.v"},
            {"name": "engine"},
        ]
        modules = {}
        for name, path in discover_block_rtl(str(tmp_path), blocks).items():
            mod = parse_verilog_ports(path)
            if mod.name:
                modules[name] = mod

        assert len(modules) == 2
        assert detect_wrapper_block(modules) == "user_project_wrapper_io"
        # And the pre-fix state, for contrast: without the pad block there is
        # no Caravel boundary to find and the flow silently takes the LLM path.
        assert detect_wrapper_block(
            {k: v for k, v in modules.items() if k != "user_project_wrapper_io"}
        ) is None


# ---------------------------------------------------------------------------
# The fix has to be reachable from the caller that actually exists.
# ---------------------------------------------------------------------------

#: EXACTLY the keys pipeline_graph builds for a passing block (~:4913). Written
#: out rather than paraphrased: the first version of the rtl_target fix passed
#: its unit test and was dead on the production path, because the test invented a
#: dict containing rtl_target and the real caller never produces one. This branch
#: has that same defect recorded for the chip_top gate sim ("its unit test passed
#: by injecting the value itself"), so the shape is pinned here on purpose.
_PRODUCTION_RESULT_KEYS = (
    "name", "success", "attempts", "gate_count", "synth_success",
    "constraints_learned", "step_log_paths", "phase",
)


def _production_result(name):
    return {"name": name, "success": True, "attempts": 1, "gate_count": 0,
            "synth_success": True, "constraints_learned": 0,
            "step_log_paths": {}, "phase": "rtl"}


class TestTheProductionCallerCanActuallyResolveRtlTarget:
    def test_the_result_dict_really_has_no_rtl_target(self):
        """Guards the premise. If a future change starts putting rtl_target in
        the block result, this test should fail and the join becomes redundant --
        which is information, not breakage."""
        r = _production_result("blk")
        assert set(r) == set(_PRODUCTION_RESULT_KEYS)
        assert "rtl_target" not in r and "rtl_path" not in r

    def test_discovery_drops_the_odd_named_block_without_the_join(self, tmp_path):
        """The live failure, reproduced from the production dict shape."""
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "user_project_wrapper.v").write_text(PAD_ADAPTER)
        _leaf(tmp_path / "rtl" / "core" / "engine.v", "engine")
        results = [_production_result("user_project_wrapper_io"),
                   _production_result("engine")]
        found = discover_block_rtl(str(tmp_path), results)
        assert "engine" in found                      # filename convention saves it
        assert "user_project_wrapper_io" not in found  # ...and cannot save this one

    def test_merging_the_spec_makes_it_resolvable(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import merge_block_specs
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "user_project_wrapper.v").write_text(PAD_ADAPTER)
        _leaf(tmp_path / "rtl" / "core" / "engine.v", "engine")
        results = [_production_result("user_project_wrapper_io"),
                   _production_result("engine")]
        queue = [{"name": "user_project_wrapper_io",
                  "rtl_target": "rtl/user_project_wrapper.v"},
                 {"name": "engine", "rtl_target": "rtl/core/engine.v"}]
        found = discover_block_rtl(
            str(tmp_path), merge_block_specs(results, queue))
        assert set(found) == {"user_project_wrapper_io", "engine"}

    def test_the_result_wins_over_the_spec(self, tmp_path):
        """A block reported what it actually did; the spec says what it should
        be. Never let the spec overwrite the report."""
        from orchestrator.langgraph.integration_helpers import merge_block_specs
        merged = merge_block_specs(
            [{"name": "blk", "success": True, "attempts": 3}],
            [{"name": "blk", "attempts": 1, "rtl_target": "rtl/blk.v"}])
        assert merged[0]["attempts"] == 3
        assert merged[0]["rtl_target"] == "rtl/blk.v"

    def test_a_block_with_no_spec_survives_the_join(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import merge_block_specs
        merged = merge_block_specs([_production_result("orphan")], [])
        assert merged[0]["name"] == "orphan"
