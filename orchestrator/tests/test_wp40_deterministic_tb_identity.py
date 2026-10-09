"""WP-40: deterministic TB reuse by content hash; the infallibility prose is gone."""
from __future__ import annotations

import inspect
from pathlib import Path

from orchestrator.langgraph import pipeline_graph as pg


def test_flags_record_the_tb_hash(tmp_path):
    tb = tmp_path / "tb.py"
    tb.write_text("print('engine')\n")
    flags = pg._tb_writer_flags({"deterministic_bfm": True, "contract": {"x": 1},
                                 "testbench_path": str(tb)})
    assert flags["tb_sha256"] == pg._file_sha256(str(tb))
    tb.write_text("print('edited')\n")
    assert pg._file_sha256(str(tb)) != flags["tb_sha256"]
    # a recorded hash is carried forward untouched on reuse
    again = pg._tb_writer_flags({"deterministic_bfm": True, "testbench_path": str(tb),
                                 "tb_sha256": flags["tb_sha256"]})
    assert again["tb_sha256"] == flags["tb_sha256"]


def test_llm_testbench_carries_no_hash(tmp_path):
    tb = tmp_path / "tb.py"
    tb.write_text("x")
    assert "tb_sha256" not in pg._tb_writer_flags({"testbench_path": str(tb)})


def test_reuse_path_checks_identity():
    src = inspect.getsource(pg.integration_dv_node)
    assert "deterministic_tb_reused" in src and "deterministic_tb_modified" in src
    assert "deterministic_tb_regenerated" not in src


def test_prompts_drop_the_infallibility_claim():
    root = Path(pg.__file__).resolve().parent.parent
    ca = (root / "langchain/prompts/contract_audit.md").read_text()
    assert "never a\nTESTBENCH_BUG" not in ca and "It is never a" not in ca
    assert "concrete BFM defect" in ca
