"""Integration validation uses Verilog module names, not logical block IDs."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.harness import top_module as tm
from orchestrator.langgraph import pipeline_graph as pg


def _install_lead_path(monkeypatch, root: Path, top_source: str):
    fabric = root / "rtl/interconnect/cs_fabric_mcufft.v"
    leaf = root / "rtl/leaf/leaf.v"
    fabric.parent.mkdir(parents=True)
    leaf.parent.mkdir(parents=True)
    fabric.write_text("module cs_fabric_mcufft(input clk); endmodule\n")
    leaf.write_text("module leaf(input clk); endmodule\n")
    rtl_paths = {"mcufft": str(fabric), "leaf": str(leaf)}

    class Lead:
        async def integrate(self, *, output_path, **_kwargs):
            Path(output_path).write_text(top_source)
            return {
                "rtl_path": output_path,
                "module_name": "chip_top",
                "mismatches": [],
                "wire_count": 2,
                "skipped_connections": [],
                "notes": "",
            }

    monkeypatch.setenv("CORESMITH_DETERMINISTIC_TOP", "0")
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "0")
    monkeypatch.setattr(pg, "_current_phase_completed", lambda *_a, **_k: [
        {"name": name, "success": True, "rtl_path": path}
        for name, path in rtl_paths.items()
    ])
    monkeypatch.setattr(pg, "discover_block_rtl", lambda *_a, **_k: rtl_paths)
    monkeypatch.setattr(pg, "load_architecture_connections", lambda *_a, **_k: ([], "chip_top"))
    monkeypatch.setattr(
        "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent", Lead
    )
    monkeypatch.setattr(pg, "lint_top_level", lambda *_a, **_k: {"clean": True, "errors": ""})
    return rtl_paths


@pytest.mark.asyncio
async def test_lead_postcondition_accepts_aliased_verilog_module(tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text("top: chip_top\n")
    _install_lead_path(
        monkeypatch,
        tmp_path,
        "module chip_top(input clk); cs_fabric_mcufft u_mcufft(clk); leaf u_leaf(clk); endmodule\n",
    )
    seen = {}

    def check(_top, expected, **_kwargs):
        seen["expected"] = set(expected)
        return None

    monkeypatch.setattr(
        "orchestrator.langchain.agents.integration_lead.assert_blocks_instantiated", check
    )

    prepared = await pg._prepare_integration_check({
        "project_root": str(tmp_path),
        "block_queue": [{"name": "mcufft"}, {"name": "leaf"}],
    })

    assert "review_bundle" in prepared
    result = prepared["review_bundle"]["integration_result"]
    assert seen["expected"] == {"cs_fabric_mcufft", "leaf"}
    assert result["block_module_names"] == {
        "mcufft": "cs_fabric_mcufft",
        "leaf": "leaf",
    }


@pytest.mark.asyncio
async def test_lead_postcondition_rejects_omitted_aliased_module(tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text("top: chip_top\n")
    _install_lead_path(
        monkeypatch,
        tmp_path,
        "module chip_top(input clk); leaf u_leaf(clk); endmodule\n",
    )
    seen = {}

    def check(_top, expected, **_kwargs):
        seen["expected"] = set(expected)
        return "missing cs_fabric_mcufft" if "cs_fabric_mcufft" in expected else None

    monkeypatch.setattr(
        "orchestrator.langchain.agents.integration_lead.assert_blocks_instantiated", check
    )
    monkeypatch.setattr(pg, "_resolve_interrupt", AsyncMock(return_value={"action": "abort"}))

    prepared = await pg._prepare_integration_check({
        "project_root": str(tmp_path),
        "block_queue": [{"name": "mcufft"}, {"name": "leaf"}],
    })

    result = prepared["integration_result"]
    assert seen["expected"] == {"cs_fabric_mcufft", "leaf"}
    assert result["postcondition_failed"] is True
    assert result["aborted"] is True


@pytest.mark.asyncio
async def test_lead_adoption_receipt_uses_aliased_verilog_module(tmp_path, monkeypatch):
    top = tmp_path / "chip_top.v"
    fabric = tmp_path / "cs_fabric_mcufft.v"
    leaf = tmp_path / "leaf.v"
    top.write_text("module chip_top; cs_fabric_mcufft u_mcufft(); leaf u_leaf(); endmodule\n")
    fabric.write_text("module cs_fabric_mcufft(); endmodule\n")
    leaf.write_text("module leaf(); endmodule\n")
    rtl_paths = {"mcufft": str(fabric), "leaf": str(leaf)}
    result = {
        "design_name": "chip_top",
        "top_module": "chip_top",
        "top_rtl_path": str(top),
        "block_rtl_paths": rtl_paths,
        "block_module_names": {"mcufft": "cs_fabric_mcufft", "leaf": "leaf"},
        "block_count": 2,
        "mismatches": [],
        "lint_clean": True,
    }
    bundle = {
        "integration_result": result,
        "agent_result": {},
        "lint_result": {"clean": True},
        "artifact_hashes": pg._integration_artifact_hashes([top, fabric, leaf]),
    }
    seen = {}

    def adopt(*_args, **kwargs):
        seen["expected"] = set(kwargs["expected_blocks"])
        return {}

    monkeypatch.setattr("orchestrator.harness.top_module.write_candidate_receipt", adopt)
    monkeypatch.setenv("CORESMITH_INTEGRATION_CHECK_PARK", "0")

    adopted = await pg._approve_integration_check({"project_root": str(tmp_path)}, bundle)

    assert adopted["integration_result"] == result
    assert seen["expected"] == {"cs_fabric_mcufft", "leaf"}


@pytest.mark.asyncio
async def test_single_block_wrapper_receipt_uses_aliased_verilog_module(tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text("chassis: none\n")
    fabric = tmp_path / "rtl/interconnect/cs_fabric_mcufft.v"
    fabric.parent.mkdir(parents=True)
    fabric.write_text("module cs_fabric_mcufft(input clk); endmodule\n")
    monkeypatch.setattr(pg, "_current_phase_completed", lambda *_a, **_k: [
        {"name": "mcufft", "success": True, "rtl_path": str(fabric)}
    ])
    monkeypatch.setattr(pg, "discover_block_rtl", lambda *_a, **_k: {"mcufft": str(fabric)})
    monkeypatch.setattr(pg, "load_architecture_connections", lambda *_a, **_k: ([], "chip_top"))
    monkeypatch.setattr(pg, "lint_top_level", lambda *_a, **_k: {"clean": True, "errors": ""})
    monkeypatch.setattr(
        "orchestrator.harness.hierarchy.elaborate_hierarchy",
        lambda *_a, **_k: {"cs_fabric_mcufft"},
    )

    result = await pg._prepare_integration_check({
        "project_root": str(tmp_path),
        "block_queue": [{"name": "mcufft"}],
    })

    integrated = result["integration_result"]
    receipt = tm.validated_candidate(tmp_path)
    assert integrated["single_block_wrapper"] is True
    assert receipt["expected_blocks"] == ["cs_fabric_mcufft"]
