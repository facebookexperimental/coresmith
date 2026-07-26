# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic tests for the arithmetic op-delay characterizer + PDK-char stage.

The synth/STA sweep itself needs yosys+OpenROAD (not in CI), so these test the
RTL emission, the XLS-form fit, prediction, ops_per_stage, caching, and the
unifying stage wiring -- all with the EDA layer mocked."""

from __future__ import annotations

from orchestrator.langgraph import arith_characterize as ac
from orchestrator.langgraph import pdk_characterize as pc


def test_emit_scalar_ops_are_valid_tops():
    for op in ("add", "sub", "mul", "cmp", "mux", "shift"):
        v, top = ac._emit_scalar(op, 16)
        assert top == "op_top" and "module op_top(" in v and "endmodule" in v
    # widths + operator semantics land in the body
    assert "a + b" in ac._emit_scalar("add", 8)[0]
    assert "a * b" in ac._emit_scalar("mul", 8)[0]
    assert "a < b" in ac._emit_scalar("cmp", 8)[0]
    assert "sel ? a : b" in ac._emit_scalar("mux", 8)[0]


def test_emit_sad_builds_absdiff_tree():
    v, top = ac._emit_sad(4, pix_w=8)
    assert top == "op_top"
    assert v.count(") - (") + v.count("] - ") >= 4  # an abs-diff subtract per term
    assert "d0 + d1 + d2 + d3" in v  # the reduction sum


def test_emit_lut_is_full_sbox_table():
    # lut(8) is a full 256-entry combinational table = the real AES S-box.
    v, top = ac._emit_lut(8)
    assert top == "op_top"
    assert "case (a)" in v and "endcase" in v
    assert v.count("y =") >= 256          # one assign per table entry (+default)
    assert "8'd99" in v                   # AES S-box[0] = 0x63 = 99
    assert "input [7:0] a" in v and "output reg [7:0] y" in v


def test_emit_gfmul_is_unrolled_gf_multiply():
    # gfmul(8) is the russian-peasant GF(2^8) multiply: 8-deep xtime + accumulate.
    v, top = ac._emit_gfmul(8)
    assert top == "op_top"
    assert "a0 = a" in v and "p0 = b[0]" in v
    assert "a7" in v and "p7" in v        # W=8 -> chain to index 7
    assert "8'd27" in v                   # AES reduction poly 0x1B = 27
    assert "assign y = p7;" in v


def test_emit_xortree_reduces_to_parity():
    v, top = ac._emit_xortree(16)
    assert top == "op_top" and "assign y = ^a;" in v


def test_new_ops_in_width_sweep():
    # the crypto/codec primitives are swept over the width grid (item 1)
    assert "gfmul" in ac.WIDTH_SWEEP_OPS and "xortree" in ac.WIDTH_SWEEP_OPS
    assert 8 in ac.DEFAULT_LUT_WIDTHS      # the S-box address width must be swept


def test_predict_prices_lut_and_gfmul_from_fitted_model():
    # With lut + gfmul in the characterized model, they are PRICED (not None) --
    # the vocabulary CoreSmith was blind to (item 1 + validation).
    key = ac.pdk_hash(None)
    ac._MODEL_CACHE[key] = {"model": {
        "lut": {"a": 0.9, "b": 0.0, "c": 2.0},     # ~9.2 ns @ addr-W=8
        "gfmul": {"a": 1.2, "b": 0.0, "c": 3.0},   # ~12.6 ns @ W=8
        "xortree": {"a": 0.05, "b": 0.6, "c": 0.2},
    }}
    try:
        d_lut = ac.predict_op_delay("lut", 8)
        d_gf = ac.predict_op_delay("gfmul", 8)
        assert d_lut is not None and d_lut > 0
        assert d_gf is not None and d_gf > 0
        assert ac.is_op_characterized("lut") is True
        assert ac.is_op_characterized("gfmul") is True
    finally:
        ac._MODEL_CACHE.pop(key, None)


def test_sbox_mixcolumns_chain_forces_two_stages_at_20ns():
    # The AES round: S-box (lut@8) -> MixColumns (gfmul@8) -> AddRoundKey
    # (xortree@8). At a 20 ns period the CHAINED delay exceeds one period, so
    # the scheduler MUST split the round into >=2 stages (the within-round
    # register the old None-priced model skipped -> WNS ~= -7.31 ns).
    from orchestrator.langgraph.pipeline_scheduler import Node, schedule_dfg
    key = ac.pdk_hash(None)
    ac._MODEL_CACHE[key] = {"model": {
        "lut": {"a": 0.9, "b": 0.0, "c": 2.0},      # ~9.2 ns @8
        "gfmul": {"a": 1.2, "b": 0.0, "c": 3.0},    # ~12.6 ns @8
        "xortree": {"a": 0.05, "b": 0.6, "c": 0.2}, # ~2.4 ns @8
    }}
    try:
        # single-instance sum > 20 ns period -> cannot be one combinational cycle
        chain = (ac.predict_op_delay("lut", 8) + ac.predict_op_delay("gfmul", 8)
                 + ac.predict_op_delay("xortree", 8))
        assert chain > 20.0
        nodes = [Node("sbox", "lut", 8), Node("mix", "gfmul", 8),
                 Node("ark", "xortree", 8)]
        edges = [("sbox", "mix"), ("mix", "ark")]
        sched = schedule_dfg(nodes, edges, 20.0)  # uses the characterized model
        assert sched.depth >= 2                    # a within-round register added
        assert all(sd <= 20.0 + 1e-9 for sd in sched.stage_delay_ns)
    finally:
        ac._MODEL_CACHE.pop(key, None)


def test_fit_recovers_linear_delay():
    # delay = 0.3*W + 2.0 (no log term) -> fit should recover a~0.3, c~2.0
    pts = [ac.OpPoint("add", w, 0.3 * w + 2.0, 100.0) for w in (4, 8, 16, 32)]
    model = ac.fit_delay_model(pts)
    assert "add" in model
    assert abs(model["add"]["a"] - 0.3) < 0.05
    assert model["add"]["max_resid_ns"] < 1e-6  # exact linear fit


def test_predict_and_ops_per_stage(monkeypatch):
    # inject a model directly into the in-process cache (no file / no EDA)
    key = ac.pdk_hash(None)
    ac._MODEL_CACHE[key] = {"model": {
        "add": {"a": 0.3, "b": 0.0, "c": 2.0},
        "mul": {"a": 0.1, "b": 2.8, "c": -5.0},
    }}
    try:
        d_add16 = ac.predict_op_delay("add", 16)
        assert abs(d_add16 - (0.3 * 16 + 2.0)) < 1e-9
        # mul slower than add at 16b (the whole point)
        assert ac.predict_op_delay("mul", 16) > d_add16
        # ops_per_stage: how many add@16 (6.8ns) fit a 20ns period -> 2
        assert ac.ops_per_stage("add", 16, 20.0) == 2
        assert ac.ops_per_stage("add", 16, 5.0) == 1  # never below 1
    finally:
        ac._MODEL_CACHE.pop(key, None)


def test_unknown_op_priced_conservatively_not_none(monkeypatch):
    # An op OUTSIDE the characterized vocabulary, with the PDK characterized,
    # is NEVER timing-free: it is priced at the SLOWEST known op (a conservative
    # proxy) so the scheduler gives it at least its own stage -- not None/0.
    key = ac.pdk_hash(None)
    ac._MODEL_CACHE[key] = {"model": {
        "add": {"a": 0.3, "b": 0.0, "c": 2.0},   # 6.8 ns @16
        "mul": {"a": 0.1, "b": 2.8, "c": -5.0},  # slower @16
    }}
    try:
        d = ac.predict_op_delay("aes_sbox", 16)
        assert d is not None and d > 0
        # == the slowest known op at that width (conservative proxy)
        assert abs(d - ac.predict_op_delay("mul", 16)) < 1e-9
        # its OWN stage: ops_per_stage returns 1 for an unknown op, never None.
        assert ac.ops_per_stage("aes_sbox", 16, 20.0) == 1
        ann = ac.predict_op_delay_annotated("aes_sbox", 16)
        assert ann["known"] is False and ann["extrapolated"] is True
        assert ann["delay_ns"] is not None
    finally:
        ac._MODEL_CACHE.pop(key, None)


def test_predict_returns_none_when_uncharacterized(monkeypatch):
    # ONLY a fully-uncharacterized PDK (no model doc) yields None -- distinct
    # from an unknown op with a characterized PDK (conservative proxy above).
    ac._MODEL_CACHE.clear()
    monkeypatch.setattr(ac, "_load_model", lambda pdk=None: None)
    assert ac.predict_op_delay("add", 16) is None
    assert ac.predict_op_delay("aes_sbox", 16) is None
    assert ac.ops_per_stage("add", 16, 20.0) is None
    assert ac.ops_per_stage("aes_sbox", 16, 20.0) is None


# --------------------------------------------------------------------------
# [mem-model-fix] arith applicability companion (parity with the mem model):
# out-of-sweep widths return a flagged result; NUMERIC behavior is unchanged.
# --------------------------------------------------------------------------

def test_op_width_in_grid_companion(monkeypatch):
    key = ac.pdk_hash(None)
    ac._MODEL_CACHE[key] = {
        "model": {"add": {"a": 0.3, "b": 0.0, "c": 2.0}},
        "points": [{"op": "add", "width": w, "delay_ns": 0.3 * w + 2.0,
                    "area_um2": 1.0, "error": ""} for w in (4, 8, 16, 32)],
    }
    try:
        assert ac.op_width_in_grid("add", 16) is True      # inside the sweep
        assert ac.op_width_in_grid("add", 4) is True       # lower edge
        assert ac.op_width_in_grid("add", 32) is True      # upper edge
        assert ac.op_width_in_grid("add", 64) is False     # out-of-sweep (above)
        assert ac.op_width_in_grid("add", 2) is False      # out-of-sweep (below)
        assert ac.op_width_in_grid("mul", 16) is None      # op not swept
        # predict_op_delay numeric behavior is UNCHANGED for both in/out of grid
        assert abs(ac.predict_op_delay("add", 16) - (0.3 * 16 + 2.0)) < 1e-9
        assert abs(ac.predict_op_delay("add", 64) - (0.3 * 64 + 2.0)) < 1e-9
        ann = ac.predict_op_delay_annotated("add", 64)
        assert ann["op"] == "add" and ann["width"] == 64
        assert ann["in_grid"] is False
        assert abs(ann["delay_ns"] - (0.3 * 64 + 2.0)) < 1e-9
        ann_in = ac.predict_op_delay_annotated("add", 16)
        assert ann_in["in_grid"] is True
    finally:
        ac._MODEL_CACHE.pop(key, None)


def test_op_width_in_grid_uncharacterized(monkeypatch):
    ac._MODEL_CACHE.clear()
    monkeypatch.setattr(ac, "_load_model", lambda pdk=None: None)
    assert ac.op_width_in_grid("add", 16) is None
    ann = ac.predict_op_delay_annotated("add", 16)
    assert ann["in_grid"] is None and ann["delay_ns"] is None


def test_stage_enabled_env_toggle(monkeypatch):
    monkeypatch.delenv("CORESMITH_PDK_CHAR", raising=False)
    assert pc.stage_enabled() is False        # default OFF
    monkeypatch.setenv("CORESMITH_PDK_CHAR", "1")
    assert pc.stage_enabled() is True
    monkeypatch.setenv("CORESMITH_PDK_CHAR", "off")
    assert pc.stage_enabled() is False


def test_ensure_pdk_characterized_folds_both_and_fails_open(monkeypatch):
    # mock both halves: arith returns a model doc, memory returns a small table
    monkeypatch.setattr(pc._arith, "ensure_characterized",
                        lambda pdk=None, force=False: {
                            "pdk_hash": "deadbeef", "delay_form": "x",
                            "model": {"add": {}, "mul": {}, "sad": {}}})
    monkeypatch.setattr(pc._arith, "_cache_path",
                        lambda pdk=None: __import__("pathlib").Path("/tmp/a.json"))
    monkeypatch.setattr(pc._mem, "load_table", lambda pdk=None: [1, 2, 3])
    s = pc.ensure_pdk_characterized()
    assert s["arith"]["ops"] == ["add", "mul", "sad"]
    assert s["memory"]["rows"] == 3
    assert s["errors"] == []
    # fail-open: an arith tool error is captured, not raised
    def _boom(pdk=None, force=False):
        raise RuntimeError("yosys missing")
    monkeypatch.setattr(pc._arith, "ensure_characterized", _boom)
    s2 = pc.ensure_pdk_characterized()
    assert any("arith" in e for e in s2["errors"])  # captured, no exception
