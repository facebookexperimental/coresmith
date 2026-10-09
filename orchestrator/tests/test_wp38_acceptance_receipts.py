"""WP-38/WP-51: the acceptance stage never passes a partial receipt and never guesses
a case setting; the benchmark-shaped stream harness is gone."""
from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

from orchestrator.langgraph import acceptance_dv as ad
from orchestrator.tests.candidate_fixtures import adopt

AXIS_TOP = """
module s_top(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tlast,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire [7:0]  m_axis_tdata,
    output wire        m_axis_tlast,
    input  wire [3:0]  mode
);
endmodule
"""


MAPPING = {"payload": "data", "input_width": 8, "output_width": 8, "packing": "bytes",
           "byte_order": "little", "sidebands": {"mode": "mode"}}


def test_no_benchmark_shape_in_the_engine():
    src = open(ad.__file__, encoding="utf-8").read()
    for tok in ("stream_core", "_STREAM_TEMPLATE", "cfg_valid", "n_frames", "word_bytes"):
        assert tok not in src, tok


def test_axis_shape_maps_payload_and_sidebands():
    c = ad.classify_contract(ad.discover_ports(AXIS_TOP))
    assert c and c["kind"] == "axis" and "mode" in c["sidebands"]
    m = ad.map_stimulus({"data": b"\x01\x02", "mode": 3}, c, mapping=MAPPING)
    assert m["payload"] == [1, 2] and m["sidebands"] == {"mode": 3}


def test_read_results_flags_truncation(tmp_path):
    p = tmp_path / "out.bin"
    p.write_bytes(struct.pack("<III", 0, 5, 4) + b"ab")          # says 4 bytes, has 2
    r = ad._read_results(p, 2)
    assert len(r) == 1 and r[0]["truncated"] is True


def test_call_accept_passes_case_when_wanted():
    seen = {}

    def with_case(expected, observed, case=None):
        seen["case"] = case
        return True

    assert ad._call_accept(with_case, b"a", b"a", "c1", {"floor": 33}) is True
    assert seen["case"] == {"name": "c1", "stimulus": {"floor": 33}}
    assert ad._call_accept(lambda e, o: e == o, b"a", b"a", "c2", {}) is True


def _project(tmp_path, cases_src: str):
    root = tmp_path / "proj"
    (root / "inputs").mkdir(parents=True)
    (root / "rtl").mkdir()
    top = root / "rtl" / "s_top.v"
    top.write_text(AXIS_TOP)
    (root / "inputs" / "acceptance_stimulus.py").write_text(cases_src + f"\nAXIS_MAPPING = {MAPPING!r}\n")
    (root / "inputs" / "golden.py").write_text("def run(stim):\n    return bytes(stim['data'])\n")
    (root / "inputs" / "accept.py").write_text("def accept(e, o):\n    return bytes(e) == bytes(o)\n")
    adopt(root, top, name="s_top")
    return root, top


def _env(monkeypatch, root):
    monkeypatch.setenv("CORESMITH_SOURCE_ROOT", str(root / "inputs" / "golden.py"))
    monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "run")
    monkeypatch.setenv("CORESMITH_FUNCTIONAL_ACCEPTANCE", str(root / "inputs" / "accept.py"))
    monkeypatch.delenv("CORESMITH_ACCEPTANCE_STIMULUS", raising=False)
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    monkeypatch.setattr(ad.shutil, "which", lambda name: "/usr/bin/" + name)


def test_incomplete_receipt_is_not_a_pass(tmp_path, monkeypatch):
    """Review round 2: two cases requested, one record returned -> passed=True."""
    root, top = _project(tmp_path,
                         "cases = [('one', {'data': b'\\x01', 'mode': 1}), ('two', {'data': b'\\x02', 'mode': 1})]\n")
    _env(monkeypatch, root)
    monkeypatch.setattr(ad, "_build", lambda *a, **k: "/bin/true")

    def fake_run(argv, **kw):
        Path(argv[2]).write_bytes(struct.pack("<III", 0, 1, 1) + b"\x01")   # ONE record
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(ad, "run_process", fake_run)
    res = ad.run_acceptance_dv(str(root), str(top), [])
    assert res["passed"] is False and res["skipped"] is False
    assert res["oracle_incomplete"] is True and res["kind"] == "oracle_incomplete"
    assert "1 complete case record(s) for 2 requested" in res["reason"]


def test_build_failure_parks_not_skips(tmp_path, monkeypatch):
    root, top = _project(tmp_path, "cases = [('one', {'data': b'\\x01', 'mode': 1})]\n")
    _env(monkeypatch, root)
    monkeypatch.setattr(ad, "_build", lambda *a, **k: None)
    res = ad.run_acceptance_dv(str(root), str(top), [])
    assert res["oracle_incomplete"] and res["kind"] == "infrastructure_error" and not res["skipped"]


def test_too_many_or_duplicate_cases_never_truncate(tmp_path, monkeypatch):
    root, top = _project(tmp_path, "cases = [('a', {'data': b'\\x01'}), ('a', {'data': b'\\x02'})]\n")
    _env(monkeypatch, root)
    res = ad.run_acceptance_dv(str(root), str(top), [])
    assert res["kind"] == "adapter_defect" and "duplicate" in res["reason"]
    monkeypatch.setenv("CORESMITH_ACCEPTANCE_MAX_CASES", "1")
    res = ad.run_acceptance_dv(str(root), str(top), [])
    assert res["kind"] == "adapter_defect" and "exceed" in res["reason"]


def test_park_payload_has_no_fix_tb_and_types_oracle_problems():
    from orchestrator.langgraph import pipeline_graph as pg
    src = open(pg.__file__, encoding="utf-8").read()
    assert '"supported_actions": (["retry", "abort"] if _acc_oracle' in src
    assert "ACCEPTANCE_ORACLE_INCOMPLETE" in src
