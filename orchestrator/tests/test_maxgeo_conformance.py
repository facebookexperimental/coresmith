# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MAX-GEOMETRY coverage for the deterministic QSPI conformance testbench.

Background: on the raster validation run the conformance TB pushed 23,380 ns of
real bus traffic through the chip and was then vetoed by the MAX-GEOMETRY gate,
because it advertised no ``# MAXGEO`` marker at all. It could not: it is
compute-lane INDEPENDENT (no golden model), so ``frame_width=64`` and
``num_triangles=64`` are not things it can drive. But ``qspi_address=16777215``,
``in_write_length=2048`` and ``out_read_length=4096`` ARE -- they are bus
dimensions with real index widths behind them.

Two properties are tested here, and they pull in opposite directions on purpose:

  * the TB now DRIVES every bus maximum it can, and marks exactly those;
  * it never marks anything else -- a marker for coverage that does not exist is
    worse than a missing marker, and the gate would silently accept it forever.
"""

from __future__ import annotations

import json
import re

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.bfm_lib import (
    QSPIContract,
    bus_maxgeo_coverage,
    classify_dim_role,
    render_conformance_tb,
    write_qspi_conformance_tb,
)
from orchestrator.langgraph.bfm_lib import maxgeo as _maxgeo

# The raster validation run's actual declared maxima (see
# coresmith-runs/exp-raster-validate-20260730). Five are bus dimensions; the
# other thirteen are compute-lane geometry.
RASTER_DIMS = {
    "frame_width": 64, "frame_height": 64,
    "framebuffer_depth": 4096, "framebuffer_width": 8,
    "zbuffer_depth": 4096, "zbuffer_width": 9,
    "triangle_store_records": 64, "triangle_record_width": 64,
    "num_triangles": 64, "screen_x": 63, "screen_y": 63,
    "triangle_depth": 255, "triangle_color": 255,
    "qspi_address": 16777215, "in_write_length": 2048,
    "out_read_length": 4096, "qspi_transaction_nibble_count": 8192,
    "qspi_command": 255,
}
BUS_DIMS = {
    "qspi_address": 16777215, "in_write_length": 2048,
    "out_read_length": 4096, "qspi_transaction_nibble_count": 8192,
    "qspi_command": 255,
}

# The gate's own parser, replicated so a change to either side is visible here.
_MARKER_RE = re.compile(r"#\s*MAXGEO\b(.*)", re.IGNORECASE)
_PAIR_RE = re.compile(r"([A-Za-z_][\w./\-]*)\s*=\s*(\d+)")


def _gate_reads(tb_src: str) -> dict:
    pairs: dict = {}
    for line in tb_src.splitlines():
        m = _MARKER_RE.search(line)
        if m:
            for name, val in _PAIR_RE.findall(m.group(1)):
                pairs[name] = int(val)
    return pairs


# ---------------------------------------------------------------------------
# Role classification -- bus vocabulary lives in the protocol layer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,role", [
    ("qspi_address", _maxgeo.ROLE_ADDRESS),
    ("bus_addr", _maxgeo.ROLE_ADDRESS),
    ("in_write_length", _maxgeo.ROLE_WRITE_LEN),
    ("input_burst_bytes", _maxgeo.ROLE_WRITE_LEN),
    ("out_read_length", _maxgeo.ROLE_READ_LEN),
    ("output_read_size", _maxgeo.ROLE_READ_LEN),
    ("qspi_transaction_nibble_count", _maxgeo.ROLE_NIBBLES),
    ("qspi_command", _maxgeo.ROLE_COMMAND),
    ("max_opcode", _maxgeo.ROLE_COMMAND),
])
def test_bus_roles_recognised(name, role):
    assert classify_dim_role(name) == role


@pytest.mark.parametrize("name", [
    "frame_width", "frame_height", "num_triangles", "screen_x",
    "framebuffer_depth", "zbuffer_width", "triangle_record_width",
    "triangle_store_records", "", "cmd_fifo_depth_of_the_compute_lane",
])
def test_non_bus_names_are_not_claimed(name):
    """An unrecognised name is NOT a bus dim. The default has to be 'we did not
    drive it', or the generator starts inventing coverage.

    ``cmd_fifo_depth_of_the_compute_lane`` is the trap: it carries the bus token
    ``cmd``, but it is a FIFO depth in the compute lane. A length token in the
    name settles it, so it is not claimed as a maximum opcode."""
    assert classify_dim_role(name) == "", name


# ---------------------------------------------------------------------------
# Coverage split
# ---------------------------------------------------------------------------
def test_raster_dims_split_bus_from_compute_lane():
    cov = bus_maxgeo_coverage(QSPIContract(), RASTER_DIMS)
    assert cov.covered == BUS_DIMS
    assert set(cov.uncovered) == set(RASTER_DIMS) - set(BUS_DIMS)
    assert cov.address == 16777215
    assert cov.write_bytes == 2048
    assert cov.read_bytes == 4096
    assert cov.opcode == 255


def test_address_beyond_the_contract_address_space_is_not_claimed():
    c = QSPIContract(addr_bytes=2)          # 16-bit space
    cov = bus_maxgeo_coverage(c, {"qspi_address": 16777215})
    assert cov.covered == {}
    assert cov.uncovered == {"qspi_address": 16777215}
    assert cov.address == 0


def test_burst_beyond_the_sim_cap_is_not_claimed(monkeypatch):
    monkeypatch.setenv(_maxgeo.MAX_BURST_ENV, "512")
    cov = bus_maxgeo_coverage(QSPIContract(), {"out_read_length": 4096})
    assert cov.covered == {}
    assert cov.read_bytes == 0


def test_write_burst_beyond_the_in_aperture_is_not_claimed():
    # IN at 0x1000, OUT at 0x2000 -> a legal IN write burst is at most 4096 B.
    cov = bus_maxgeo_coverage(QSPIContract(), {"in_write_length": 8192})
    assert cov.covered == {}
    assert cov.write_bytes == 0


def test_nibble_count_claimed_only_when_it_matches_real_traffic():
    c = QSPIContract()
    # 4096-byte read == 8192 data nibbles -> claimed
    ok = bus_maxgeo_coverage(c, {"out_read_length": 4096, "nibble_count": 8192})
    assert ok.covered == {"out_read_length": 4096, "nibble_count": 8192}
    # a nibble count that does NOT correspond to the longest driven burst is a
    # claim about traffic that never happened
    bad = bus_maxgeo_coverage(c, {"out_read_length": 4096, "nibble_count": 9999})
    assert bad.covered == {"out_read_length": 4096}
    assert bad.uncovered == {"nibble_count": 9999}
    # ...and with no burst at all there is nothing to count nibbles of
    alone = bus_maxgeo_coverage(c, {"nibble_count": 8192})
    assert alone.covered == {}


def test_no_declared_dims_covers_nothing():
    cov = bus_maxgeo_coverage(QSPIContract(), {})
    assert cov.covered == {} and cov.uncovered == {}


# ---------------------------------------------------------------------------
# The generated testbench
# ---------------------------------------------------------------------------
def test_generated_tb_drives_and_marks_exactly_the_bus_maxima():
    src = render_conformance_tb(QSPIContract(), "chip_top", declared_dims=RASTER_DIMS)
    compile(src, "<conformance_tb>", "exec")
    assert _gate_reads(src) == BUS_DIMS
    # each marked maximum has REAL traffic behind it
    assert "await bfm.write(0xFFFFFF," in src
    assert "_n = 2048" in src
    assert "await bfm.read(c.out_addr, 4096)" in src
    assert "await bfm.send_opcode_only(0xFF)" in src


def test_uncovered_dims_are_recorded_but_never_read_as_coverage():
    """`# MAXGEO-UNCOVERED` (hyphen) WOULD match the gate's `#\\s*MAXGEO\\b`
    regex and turn a confession into a coverage claim for every compute-lane
    dim. The underscore form cannot."""
    src = render_conformance_tb(QSPIContract(), "chip_top", declared_dims=RASTER_DIMS)
    assert "# MAXGEO_NOT_COVERED: " in src
    assert "frame_width=64" in src                       # recorded...
    assert "frame_width" not in _gate_reads(src)         # ...but never claimed
    assert "MAXGEO-UNCOVERED" not in src
    assert "# MAXGEO_SCOPE:" in src


def test_no_declared_dims_keeps_the_tb_byte_identical():
    c = QSPIContract()
    assert render_conformance_tb(c, "chip_top") == render_conformance_tb(
        c, "chip_top", declared_dims=None)
    assert "MAXGEO" not in render_conformance_tb(c, "chip_top")


def test_writer_reports_coverage_and_carries_the_contract(tmp_path):
    out = tmp_path / "tb" / "integration" / "test_chip_top.py"
    res = write_qspi_conformance_tb(
        str(tmp_path), "chip_top", QSPIContract(), str(out),
        declared_dims=RASTER_DIMS,
    )
    assert res["maxgeo_covered"] == BUS_DIMS
    assert set(res["maxgeo_uncovered"]) == set(RASTER_DIMS) - set(BUS_DIMS)
    # the CONTRACT travels with the result so a downstream gate can recompute
    # what this TB was capable of, instead of trusting what it claimed
    assert res["contract"]["addr_bytes"] == 3
    assert out.is_file()


# ---------------------------------------------------------------------------
# The gate's scoped verdict
# ---------------------------------------------------------------------------
def _project(tmp_path, dims: dict) -> str:
    cs = tmp_path / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    (cs / "ers_spec.json").write_text(json.dumps({"ers": {"constraints": [
        {"name": n, "max": v} for n, v in dims.items()
    ]}}), encoding="utf-8")
    return str(tmp_path)


def _conformance_tb(tmp_path, dims: dict) -> tuple[str, dict]:
    out = tmp_path / "tb" / "integration" / "test_chip_top.py"
    res = write_qspi_conformance_tb(
        str(tmp_path), "chip_top", QSPIContract(), str(out), declared_dims=dims)
    return str(out), res


def test_gate_scopes_the_deterministic_conformance_tb(tmp_path):
    pr = _project(tmp_path, RASTER_DIMS)
    tb, res = _conformance_tb(tmp_path, RASTER_DIMS)
    v = pipeline_graph._maxgeo_gate_verdict(pr, tb, res)
    assert v is not None and v["advisory"] is True
    assert v["bus_covered"] == BUS_DIMS
    # the compute-lane remainder is REPORTED, not dropped
    assert "frame_width" in v["uncovered_dims"]
    assert "NOT COVERED" in v["reason"]
    assert "no compute oracle" in v["reason"].lower()


def test_gate_still_bites_when_the_generator_skips_a_bus_maximum(tmp_path):
    """The relaxation recomputes the expected bus coverage from the CONTRACT.
    A codegen regression that stops driving the max read burst is a hard fail --
    the TB does not get to mark its own homework."""
    pr = _project(tmp_path, RASTER_DIMS)
    tb, res = _conformance_tb(tmp_path, RASTER_DIMS)
    text = open(tb).read().replace("out_read_length=4096 ", "")
    open(tb, "w").write(text)
    v = pipeline_graph._maxgeo_gate_verdict(pr, tb, res)
    assert v is not None and v.get("advisory") is not True
    assert v["bus_skipped"] == {"out_read_length": 4096}


def test_gate_does_not_scope_an_llm_authored_tb(tmp_path):
    """Same testbench TEXT, but the caller's record does not identify it as the
    engine's deterministic conformance TB -> the gate is unchanged. The
    discriminator is the caller's record precisely so a generated file cannot
    talk its way into the relaxation."""
    pr = _project(tmp_path, RASTER_DIMS)
    tb, _res = _conformance_tb(tmp_path, RASTER_DIMS)
    for record in (None, {}, {"deterministic_bfm": True}, {"conformance_only": True}):
        v = pipeline_graph._maxgeo_gate_verdict(pr, tb, record)
        assert v is not None and v.get("advisory") is not True, record


def test_gate_does_not_scope_without_the_scope_marker_in_the_artifact(tmp_path):
    pr = _project(tmp_path, RASTER_DIMS)
    tb, res = _conformance_tb(tmp_path, RASTER_DIMS)
    text = open(tb).read().replace("# MAXGEO_SCOPE:", "# (scope line removed)")
    open(tb, "w").write(text)
    v = pipeline_graph._maxgeo_gate_verdict(pr, tb, res)
    assert v is not None and v.get("advisory") is not True


def test_scope_can_be_switched_off(tmp_path, monkeypatch):
    pr = _project(tmp_path, RASTER_DIMS)
    tb, res = _conformance_tb(tmp_path, RASTER_DIMS)
    assert pipeline_graph._maxgeo_gate_verdict(pr, tb, res)["advisory"] is True
    monkeypatch.setenv("CORESMITH_MAXGEO_CONFORMANCE_SCOPE", "0")
    v = pipeline_graph._maxgeo_gate_verdict(pr, tb, res)
    assert v is not None and v.get("advisory") is not True


def test_a_bus_only_design_gets_a_clean_pass_not_an_advisory(tmp_path):
    """When every declared maximum IS a bus maximum, the conformance TB covers
    them all and the gate returns a plain pass -- the scope path is not
    involved."""
    pr = _project(tmp_path, BUS_DIMS)
    tb, res = _conformance_tb(tmp_path, BUS_DIMS)
    assert pipeline_graph._maxgeo_gate_verdict(pr, tb, res) is None
