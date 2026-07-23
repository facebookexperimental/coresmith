# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for complexity-aware block decomposition (block_complexity).

Fully deterministic -- no LLM, no EDA. The estimator + decomposer are pure
AST analysis over a golden reference. Fixtures are small synthetic goldens so
the tests do not depend on any particular design's golden.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.langgraph.block_complexity import (
    MODELING_ALGO_THRESHOLD,
    MODELING_LOC_THRESHOLD,
    estimate_block_complexity,
    propose_decomposition,
    resolve_block_slice,
    _parse_functions,
)


# ---------------------------------------------------------------------------
# Fixtures: a FAT golden (many algorithms, high LOC, cross-function state) and
# a THIN golden (one tiny function) that share the block_complexity's public
# entry-hint keys ("intra_rd", "bytestream", "output_byte_fifo").
# ---------------------------------------------------------------------------

def _fat_golden(tmp_path: Path) -> str:
    """A synthetic 'intra_rd' golden: transform + quant + reconstruct + pred +
    rd-cost + chroma, all fused, with arrays crossing function boundaries."""
    lines = ["import numpy as np", ""]

    def emit(name: str, body_lines: list[str]) -> None:
        lines.append(f"def {name}(a, b, qp):")
        lines.extend("    " + ln for ln in body_lines)
        lines.append("")

    # transform kernel (distinct algorithm: transform)
    emit("_fdct4", ["d = a + b", "for i in range(4):", "    d = d * 2 + b",
                    "return d"])
    emit("fdct_quant", ["w = _fdct4(a, b, qp)", "return quantize(w, b, qp)"])
    # quant (distinct algorithm: quant)
    emit("quantize", ["r = a", "for i in range(16):",
                      "    r = (r * 13 + 8) >> 4", "return r"])
    emit("dequantize", ["return a * qp + 1"])
    # reconstruct (distinct algorithm: reconstruct) + clip
    emit("reconstruct", ["p = dequantize(a, b, qp)", "recY[0] = clip255(p)",
                         "return recY[0]"])
    emit("clip255", ["return 0 if a < 0 else (255 if a > 255 else a)"])
    # intra prediction (distinct algorithm: intra_pred). Reads recY (the
    # reconstructed-neighbour array written by reconstruct/_encode_mb) but never
    # writes it -> a genuine cross-function data flow the locality axis counts.
    emit("pred_4x4", ["m = avail_modes_4x4(a, b, qp)", "nbr = recY[0]",
                      "for i in range(9):", "    p = a + nbr + i", "return p"])
    emit("avail_modes_4x4", ["return [0, 1, 2] if a else [2]"])
    # rd cost (distinct algorithm: rd_cost)
    emit("_rd_cost", ["ssd = a * a", "bits = _residual_bits(b, qp)",
                      "return ssd + ((bits * qp) >> 8)"])
    emit("_residual_bits", ["n = 0", "for c in range(16):", "    n += c",
                            "return n"])
    # mode decision that fuses everything (the monolith entry)
    encode_mb = ["recY = np.zeros((16, 16))"]
    for _ in range(40):  # inflate LOC well past the threshold
        encode_mb.append("t = fdct_quant(a, b, qp)")
        encode_mb.append("r = reconstruct(t, b, qp)")
        encode_mb.append("p = pred_4x4(r, b, qp)")
        encode_mb.append("c = _rd_cost(p, b, qp)")
        encode_mb.append("if c > 0:")
        encode_mb.append("    recY[0] = c")
    encode_mb.append("return recY")
    emit("_encode_mb", encode_mb)

    src = "\n".join(lines)
    p = tmp_path / "golden.py"
    p.write_text(src)
    return str(p)


def _thin_golden(tmp_path: Path) -> str:
    """A synthetic 'output_byte_fifo' golden: one trivial helper."""
    src = (
        "def _as_yuv(frame):\n"
        "    y = frame[0]\n"
        "    return y, y, y\n"
    )
    p = tmp_path / "golden_thin.py"
    p.write_text(src)
    return str(p)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class TestEstimator:
    def test_fat_block_flagged_over_budget(self, tmp_path):
        g = _fat_golden(tmp_path)
        est = estimate_block_complexity("intra_rd_encode_core", g,
                                        spec_text="flip_flop_budget = 26000 FF")
        assert est["over_budget"] is True
        assert est["loc"] > MODELING_LOC_THRESHOLD
        assert est["distinct_algorithms"] > MODELING_ALGO_THRESHOLD
        # the breach messages name the modeling-complexity axis (the wall)
        assert any("modeling_complexity" in b for b in est["axis_breaches"])

    def test_thin_block_under_budget(self, tmp_path):
        g = _thin_golden(tmp_path)
        est = estimate_block_complexity("output_byte_fifo", g,
                                        spec_text="flip_flop_budget = 2500 FF")
        assert est["over_budget"] is False
        assert est["axis_breaches"] == []
        assert est["loc"] <= MODELING_LOC_THRESHOLD

    def test_distinct_algorithm_count(self, tmp_path):
        g = _fat_golden(tmp_path)
        est = estimate_block_complexity("intra_rd_encode_core", g)
        # transform, quant, reconstruct, intra_pred, rd_cost all present
        for algo in ("transform", "quant", "reconstruct", "intra_pred", "rd_cost"):
            pass  # algorithm names are internal; assert the count captured them
        assert est["distinct_algorithms"] >= 5

    def test_data_locality_counts_cross_function_arrays(self, tmp_path):
        g = _fat_golden(tmp_path)
        est = estimate_block_complexity("intra_rd_encode_core", g)
        # recY is written in _encode_mb + reconstruct and read across functions
        assert est["locality"] >= 1

    def test_latency_scales_with_ops(self, tmp_path):
        g = _fat_golden(tmp_path)
        est = estimate_block_complexity("intra_rd_encode_core", g)
        assert est["latency_cyc"] > 0

    def test_ff_budget_parsed(self, tmp_path):
        g = _fat_golden(tmp_path)
        est = estimate_block_complexity("intra_rd_encode_core", g,
                                        spec_text="flip_flop_budget = 26,000 FF")
        assert est["ff_budget"] == 26000


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class TestDecomposer:
    def test_fat_block_splits_into_multiple_subblocks(self, tmp_path):
        g = _fat_golden(tmp_path)
        subs = propose_decomposition("intra_rd_encode_core", g)
        assert len(subs) > 1

    def test_subblocks_cover_whole_slice_without_overlap(self, tmp_path):
        g = _fat_golden(tmp_path)
        stats = _parse_functions(Path(g).read_text())
        slice_fns = set(resolve_block_slice("intra_rd_encode_core", stats))
        subs = propose_decomposition("intra_rd_encode_core", g)
        assigned: list[str] = []
        for s in subs:
            assigned.extend(s["golden_functions"])
        # non-overlapping
        assert len(assigned) == len(set(assigned)), "sub-blocks overlap"
        # full coverage of the block's golden slice
        assert set(assigned) == slice_fns

    def test_subblocks_have_interface_contracts(self, tmp_path):
        g = _fat_golden(tmp_path)
        subs = propose_decomposition("intra_rd_encode_core", g)
        # at least one named sub-block emits an AXI-Stream contract with i/o
        named = {s["sub_block"] for s in subs}
        assert any("transform_quant" in n for n in named)
        for s in subs:
            ic = s["interface_contract"]
            assert "inputs" in ic and "outputs" in ic

    def test_thin_block_stays_whole(self, tmp_path):
        g = _thin_golden(tmp_path)
        subs = propose_decomposition("output_byte_fifo", g)
        # no known monolith seed table -> single passthrough partition
        assert len(subs) == 1
        assert subs[0]["sub_block"] == "output_byte_fifo"

    def test_mincut_prefers_low_cross_traffic_boundaries(self, tmp_path):
        """The transform kernels (_fdct4/quantize/dequantize/fdct_quant) share
        state + call edges and must land in the transform_quant group, NOT be
        scattered across mode_decision/reconstruct -- i.e. the cut runs where
        cross-traffic is least."""
        g = _fat_golden(tmp_path)
        subs = propose_decomposition("intra_rd_encode_core", g)
        by_name = {s["sub_block"]: set(s["golden_functions"]) for s in subs}
        tq = next(v for k, v in by_name.items() if k.endswith("transform_quant"))
        # the tightly-coupled transform/quant kernels are together
        assert {"_fdct4", "fdct_quant", "quantize"} <= tq
        # and are NOT split into a different group
        for name, fns in by_name.items():
            if name.endswith("transform_quant"):
                continue
            assert "_fdct4" not in fns
            assert "fdct_quant" not in fns


# ---------------------------------------------------------------------------
# AST parsing sanity
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_captures_calls_and_arrays(self, tmp_path):
        g = _fat_golden(tmp_path)
        stats = _parse_functions(Path(g).read_text())
        assert "_encode_mb" in stats
        emb = stats["_encode_mb"]
        assert "fdct_quant" in emb.calls
        assert "recY" in emb.array_writes
        assert emb.cyclomatic > 1  # has if/for branches

    def test_syntax_error_golden_returns_empty(self):
        from orchestrator.langgraph.block_complexity import _parse_functions
        assert _parse_functions("def broken(:\n  pass") == {}
