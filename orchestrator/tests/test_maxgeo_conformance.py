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
from pathlib import Path

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
    v = pipeline_graph._maxgeo_gate_verdict(pr, tb, res)
    # run3-followups: full coverage is an explicit PASS verdict, not None.
    assert v is not None and v.get("verdict") == "pass"


# ---------------------------------------------------------------------------
# Generator-side MAX-CONFIGURATION functional case (run3-followups #1)
# ---------------------------------------------------------------------------
#
# The deterministic codegen baked exactly ONE host-flow case -- the FIRST
# acceptance case, which is the small canonical KAT. It therefore proved nothing
# about any declared maximum, and the operator had to hand-add a max-geometry
# functional test to the generated artifact. These tests pin the generator-side
# fix, and they run it through the PRODUCTION writer
# (``write_deterministic_integration_tb``), not through a test-shaped call.

# A generic two-parameter design: one small canonical case and one larger one.
# Names are placeholders on purpose -- the selection rule is over MAGNITUDES.
_GEN_GOLDEN = '''
def run(stimulus):
    n = int(stimulus["unit_count"])
    data = bytes(stimulus["data"])
    # two output bytes per input byte -> OUT extent differs from IN extent
    return bytes(v for b in data for v in (((b + n) & 0xFF), ((b ^ n) & 0xFF)))
'''

_GEN_ACCEPT = '''
def _case(n):
    return {"unit_count": n, "cfg0": n, "data": bytes((i * 7) & 0xFF for i in range(16 * n))}

cases = [
    ("canonical_smoke", _case(1)),
    ("mid", _case(4)),
    ("max_config", _case(8)),
    ("mid2", _case(2)),
]
'''

# What such a design declares. ``unit_count`` is the only compute-lane dim a
# stimulus scalar names AND drives at its maximum.
_GEN_DIMS = {
    "unit_count": 8,
    "internal_store_depth": 128,
    "qspi_address": 16777215,
    "in_write_length": 128,
    "out_read_length": 256,
}


def _generic_run(tmp_path, *, dims: dict | None = None) -> str:
    root = tmp_path / "run"
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "design_golden.py").write_text(_GEN_GOLDEN, encoding="utf-8")
    (root / "inputs" / "acceptance_stimulus.py").write_text(
        _GEN_ACCEPT, encoding="utf-8")
    (root / ".coresmith" / "ers_spec.json").write_text(json.dumps({"ers": {
        "parameters": [
            {"name": n, "role": "dimension", "max": v}
            for n, v in (dims if dims is not None else _GEN_DIMS).items()
        ]
    }}), encoding="utf-8")
    return str(root)


def _write_integration_tb(pr: str) -> tuple[str, dict]:
    """The PRODUCTION path: plan -> writer -> artifact on disk."""
    from orchestrator.langgraph.bfm_lib import (
        build_plan_from_run,
        declared_dimensional_maxima,
        write_deterministic_integration_tb,
    )
    c = QSPIContract()
    plan = build_plan_from_run(pr, c)
    assert plan is not None, "the generic run must yield a host-flow plan"
    out = str(Path(pr) / "tb" / "integration" / "test_chip_top.py")
    res = write_deterministic_integration_tb(
        pr, "chip_top", c, plan, out, include_conformance=True,
        declared_dims=declared_dimensional_maxima(pr),
    )
    return out, res


def test_max_config_case_is_selected_by_magnitude_not_by_order(tmp_path):
    """Generic selection: the case maximizing (IN payload bytes, cfg0) wins --
    it is neither first nor last in the declared list."""
    from orchestrator.langgraph.bfm_lib import (
        load_acceptance_cases,
        select_max_geometry_case,
    )
    pr = _generic_run(tmp_path)
    cases = load_acceptance_cases(pr)
    assert [n for n, _ in cases] == ["canonical_smoke", "mid", "max_config", "mid2"]
    assert select_max_geometry_case(cases)[0] == "max_config"


def test_writer_emits_a_second_baked_max_geometry_test(tmp_path):
    """The production writer emits a SECOND host-flow test for the max case,
    with its golden baked at GENERATION time (bytes in the file, no runtime
    import of the reference)."""
    pr = _generic_run(tmp_path)
    tb, res = _write_integration_tb(pr)
    src = Path(tb).read_text(encoding="utf-8")
    compile(src, "<integration_tb>", "exec")          # must parse

    assert res["case_name"] == "canonical_smoke"       # primary is unchanged
    assert res["maxgeo_case"]["name"] == "max_config"
    assert res["maxgeo_case"]["emitted_test"] is True
    assert res["test_count"] == 3                      # host-flow + max + conformance

    assert "async def deterministic_qspi_dv_max_geometry(dut):" in src
    assert "MAXGEO_EXPECTED = bytes.fromhex(" in src   # baked, not imported
    assert "import acceptance_stimulus" not in src
    assert "MAXGEO_IN_WRITES" in src


def test_max_geometry_case_marker_is_emitted_and_is_not_a_coverage_claim(tmp_path):
    """``# MAXGEO_CASE`` records WHICH case and at what magnitudes -- and the
    underscore form must not be readable as a coverage claim for dims literally
    named cfg0 / in_bytes / out_bytes."""
    pr = _generic_run(tmp_path)
    tb, _res = _write_integration_tb(pr)
    src = Path(tb).read_text(encoding="utf-8")
    assert "# MAXGEO_CASE: name=max_config cfg0=8 in_bytes=128 out_bytes=256" in src
    read = _gate_reads(src)
    for k in ("cfg0", "in_bytes", "out_bytes", "name"):
        assert k not in read, k


def test_functional_dims_are_claimed_only_with_driven_evidence(tmp_path):
    """A declared dim joins ``# MAXGEO`` only when a stimulus scalar NAMES it and
    carries its declared maximum. Everything else stays confessed -- a big case
    is not evidence for a dimension it does not drive."""
    pr = _generic_run(tmp_path)
    tb, res = _write_integration_tb(pr)
    src = Path(tb).read_text(encoding="utf-8")
    read = _gate_reads(src)

    assert res["maxgeo_functional"] == {"unit_count": 8}
    assert read["unit_count"] == 8                     # driven: cfg0/unit_count=8
    # declared, NOT driven by any scalar of the chosen case -> still confessed
    assert "internal_store_depth" not in read
    assert "internal_store_depth=128" in src
    assert "# MAXGEO_NOT_COVERED: " in src


def test_no_larger_case_means_no_second_test(tmp_path):
    """When the FIRST case already is the largest, a second identical test would
    prove nothing: no extra test is emitted, but the marker still records that
    the primary IS the max-geometry case."""
    root = tmp_path / "run"
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "design_golden.py").write_text(_GEN_GOLDEN, encoding="utf-8")
    (root / "inputs" / "acceptance_stimulus.py").write_text(
        _GEN_ACCEPT.replace('("canonical_smoke", _case(1))', '("biggest", _case(8))')
        .replace('("max_config", _case(8))', '("small", _case(1))'),
        encoding="utf-8")
    (root / ".coresmith" / "ers_spec.json").write_text(json.dumps({"ers": {
        "parameters": [{"name": n, "role": "dimension", "max": v}
                       for n, v in _GEN_DIMS.items()]}}), encoding="utf-8")
    tb, res = _write_integration_tb(str(root))
    src = Path(tb).read_text(encoding="utf-8")
    assert res["maxgeo_case"]["is_primary"] is True
    assert res["maxgeo_case"]["emitted_test"] is False
    assert res["test_count"] == 2
    assert "deterministic_qspi_dv_max_geometry" not in src
    assert "# MAXGEO_CASE: name=biggest" in src


def test_a_run_without_acceptance_cases_is_byte_identical(tmp_path):
    """No acceptance module -> no max case -> the emitted TB is exactly what it
    was before any of this existed."""
    from orchestrator.langgraph.bfm_lib import (
        StimulusPlan,
        render_integration_tb,
        write_deterministic_integration_tb,
    )
    c = QSPIContract()
    plan = StimulusPlan(cfg=[(c.cfg0_addr, 1, 4)],
                        writes=[(c.in_addr, bytes(range(16)))],
                        out_addr=c.out_addr, out_len=16,
                        expected=bytes(range(16)), case_name="unit")
    out = str(tmp_path / "tb" / "test_chip_top.py")
    res = write_deterministic_integration_tb(
        str(tmp_path), "chip_top", c, plan, out)
    assert res["maxgeo_case"] is None
    assert res["test_count"] == 1
    assert Path(out).read_text(encoding="utf-8") == render_integration_tb(
        c, "chip_top", plan)


def test_the_generated_max_geometry_tb_satisfies_the_gate(tmp_path):
    """End-to-end through the PRODUCTION gate: the artifact the writer produces
    is one the MAX-GEOMETRY gate accepts (pass or advisory), not a hard fail."""
    pr = _generic_run(tmp_path)
    tb, res = _write_integration_tb(pr)
    v = pipeline_graph._maxgeo_gate_verdict(pr, tb, res)
    assert v is None or v.get("verdict") == "pass" or v.get("advisory") is True, v


# ---------------------------------------------------------------------------
# Dimension-registry alignment (run3-followups #2)
# ---------------------------------------------------------------------------
#
# On the live run three numbers disagreed: the gate ENFORCED 9 dims, the TB
# CONFESSED 13, the declared table had 18. Nothing lied -- they were three
# derivations. One derivation now, three views of it.

def test_registry_matches_the_gate_declared_dimension_derivation(tmp_path):
    """``declared_dimensional_maxima`` IS what the gate derives today, so the
    call-site swap is a no-op on behaviour. Both the typed ``parameters`` block
    and the legacy generic harvest are exercised."""
    from orchestrator.langgraph.bfm_lib import declared_dimensional_maxima
    cs = tmp_path / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    (cs / "ers_spec.json").write_text(json.dumps({"ers": {
        "parameters": [
            {"name": "unit_count", "role": "dimension", "max": 8},
            {"name": "opcode_select", "role": "mode", "max": 255},
        ],
        # harvested generically (no parameters entry) -- must still appear
        "constraints": [{"name": "scraped_depth", "max": 512}],
    }}), encoding="utf-8")
    assert declared_dimensional_maxima(str(tmp_path)) == \
        pipeline_graph._declared_dimensions(str(tmp_path))


def test_registry_is_empty_for_a_legacy_prose_spec(tmp_path):
    from orchestrator.langgraph.bfm_lib import declared_dimensional_maxima
    assert declared_dimensional_maxima(str(tmp_path)) == {}


def test_demand_partitions_the_declared_table_and_exposes_value_collisions():
    """The 18/13/9 reconciliation, on the exact shape that produced it: five bus
    dims are PROVEN, four compute-lane dims merely COLLIDE in value with a bus
    marker (the old gate counted those as covered), nine are MISSING. 5+4+9=18,
    and the confession (value_only|missing) is 13."""
    from orchestrator.langgraph.bfm_lib import maxgeo_demand
    d = maxgeo_demand(RASTER_DIMS, BUS_DIMS)
    assert set(d.proven) == set(BUS_DIMS)
    assert len(d.proven) + len(d.value_only) + len(d.missing) == len(RASTER_DIMS)
    # 4096 and 255 are bus marker values AND compute-lane maxima
    assert d.value_only == {
        "framebuffer_depth": 4096, "triangle_color": 255,
        "triangle_depth": 255, "zbuffer_depth": 4096,
    }
    assert len(d.missing) == 9
    assert len(d.unproven) == 13
    # today's gate demands exactly `missing` -- swapping in maxgeo_demand is
    # semantics-preserving for the demand, and ADDS the weak-evidence set
    marker_values = set(BUS_DIMS.values())
    assert d.missing == {
        n: v for n, v in RASTER_DIMS.items() if v not in marker_values
    }


def test_demand_requires_name_and_value_together():
    from orchestrator.langgraph.bfm_lib import maxgeo_demand
    d = maxgeo_demand({"a": 64}, {"a": 32})     # right name, wrong value
    assert d.proven == {} and d.missing == {"a": 64}
    d2 = maxgeo_demand({"a": 64}, {"a": 64})
    assert d2.proven == {"a": 64} and not d2.unproven


def test_the_confession_and_the_classification_are_one_computation(tmp_path):
    """The generator's ``MAXGEO_NOT_COVERED`` line and the shared bus/compute
    classification cannot drift: the confession IS the classification's
    ``uncovered``, minus exactly the dims a functional case drove."""
    from orchestrator.langgraph.bfm_lib import (
        bus_maxgeo_coverage,
        declared_dimensional_maxima,
    )
    pr = _generic_run(tmp_path)
    tb, res = _write_integration_tb(pr)
    src = Path(tb).read_text(encoding="utf-8")
    dims = declared_dimensional_maxima(pr)
    cov = bus_maxgeo_coverage(QSPIContract(), dims)
    confessed = {}
    for line in src.splitlines():
        if line.startswith("# MAXGEO_NOT_COVERED:"):
            for n, v in _PAIR_RE.findall(line.split(":", 1)[1]):
                confessed[n] = int(v)
    assert confessed == {
        n: v for n, v in cov.uncovered.items()
        if n not in res["maxgeo_functional"]
    }


# ---------------------------------------------------------------------------
# Token matching -- never claim an axis you did not drive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim,key,ok", [
    ("unit_count", "unit_count", True),        # exact
    ("frame_width", "width", True),            # key tokens subset of dim tokens
    ("width", "frame_width", True),            # and the other direction
    ("frame_width", "burst_width", False),     # shared token, different axis
    ("num_triangles", "triangle_depth", False),
    ("", "unit_count", False),
])
def test_token_match_rule(dim, key, ok):
    from orchestrator.langgraph.bfm_lib import token_match
    assert token_match(dim, key) is ok


def test_functional_dims_need_the_value_too():
    from orchestrator.langgraph.bfm_lib import functional_maxgeo_dims
    dims = {"unit_count": 8, "frame_width": 64}
    assert functional_maxgeo_dims({"unit_count": 8}, dims) == {"unit_count": 8}
    # right name, BELOW the maximum -> not driven at max -> not claimed
    assert functional_maxgeo_dims({"unit_count": 4}, dims) == {}
    # right value, unrelated name -> a coincidence, not evidence
    assert functional_maxgeo_dims({"unrelated_thing": 8}, dims) == {}
