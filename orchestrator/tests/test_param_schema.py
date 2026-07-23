# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PARAM-SCHEMA-1 -- typed dimensional-parameter schema in the ERS.

The schema turns the two default-on gates whose deterministic halves used to
no-op (maxgeo `_declared_dimensions` harvested {} from prose ERS docs; the mem
manifest requirement was global-OFF for legacy specs) into fully-deterministic
gates driven off the machine-readable ERS `parameters` block.

DOMAIN-GENERIC: parameter names are DATA. The two-domain proof is the video
fixture (`frame_width`/`frame_height`) vs the NON-video AES-GCM fixture
(`max_message_blocks` dimension + `key_modes` mode).
"""

from __future__ import annotations

import json

import pytest

from orchestrator.architecture import param_schema as ps
from orchestrator.langgraph import pipeline_graph


# ===========================================================================
# Helpers + two-domain fixtures (names are data, not engine vocabulary)
# ===========================================================================
def _write_ers(root, ers_body: dict) -> None:
    cs = root / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    (cs / "ers_spec.json").write_text(
        json.dumps({"ers": ers_body}), encoding="utf-8"
    )


def _write_tb(path, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _write_spec(root, block: str, text: str) -> None:
    d = root / "arch" / "uarch_specs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{block}.md").write_text(text, encoding="utf-8")


# Domain A: video codec (structured parameters block).
_VIDEO_PARAMS = {
    "summary": "video codec",
    "parameters": [
        {"name": "frame_width", "role": "dimension", "max": 640, "unit": "pixels"},
        {"name": "frame_height", "role": "dimension", "max": 352, "unit": "pixels"},
    ],
}

# Domain B (NON-video): AES-GCM authenticated-encryption core -- a dimension
# (max_message_blocks) AND a mode (key_modes, enumerated key sizes). Zero video
# vocabulary; proves the schema + gate are domain-generic.
_AES_PARAMS = {
    "summary": "AES-GCM authenticated encryption core",
    "parameters": [
        {"name": "max_message_blocks", "role": "dimension", "max": 1024,
         "unit": "blocks"},
        {"name": "key_modes", "role": "mode", "boundary_values": [128, 192, 256]},
    ],
}

# Legacy prose ERS: dims live ONLY in prose (no parameters block, no structured
# {name,max} objects) -> both the structured read and the generic fallback
# harvest {} -> the maxgeo gate no-ops (safety: never spuriously fail a design
# that predates the schema).
_LEGACY_PROSE = {
    "summary": "streaming filter that processes frames up to 640 pixels wide",
    "functional_requirements": [
        "The core accepts frames as wide as 640 and as tall as 352.",
        "Latency under 10 us.",
    ],
}

# Legacy STRUCTURED ERS (no parameters block, but the pre-schema generic
# {name,max} harvest still applies) -> the fallback must be preserved.
_LEGACY_STRUCTURED = {
    "summary": "codec",
    "constraints": [{"name": "line_len", "max": 4096}],
}


# ===========================================================================
# 1. Schema validation (good / malformed / empty-with-statement)
# ===========================================================================
class TestSchemaValidation:
    def test_good_entry_normalizes(self):
        norm, errs = ps.normalize_parameter(
            {"name": "frame_width", "role": "dimension", "max": 640,
             "unit": "pixels"})
        assert errs == []
        assert norm["name"] == "frame_width"
        assert norm["role"] == "dimension"
        assert norm["min"] == 0            # default filled
        assert norm["max"] == 640
        assert norm["unit"] == "pixels"
        assert norm["boundary_values"][-1] == 640   # auto-computed

    def test_missing_name_is_malformed(self):
        norm, errs = ps.normalize_parameter({"role": "dimension", "max": 64})
        assert norm is None
        assert errs and "name" in errs[0]

    def test_invalid_role_is_malformed(self):
        norm, errs = ps.normalize_parameter(
            {"name": "x", "role": "wobble", "max": 64})
        assert norm is None
        assert any("role" in e for e in errs)

    def test_dimension_without_max_is_malformed(self):
        norm, errs = ps.normalize_parameter({"name": "x", "role": "dimension"})
        assert norm is None
        assert any("max" in e for e in errs)

    def test_mode_needs_no_max(self):
        norm, errs = ps.normalize_parameter(
            {"name": "key_modes", "role": "mode",
             "boundary_values": [128, 192, 256]})
        assert errs == []
        assert norm["role"] == "mode"
        assert norm["max"] is None
        assert norm["boundary_values"] == [128, 192, 256]

    def test_validate_list_drops_malformed_keeps_good(self):
        normed, errors = ps.validate_parameters([
            {"name": "good", "role": "dimension", "max": 32},
            {"name": "", "max": 8},                       # bad: no name
            {"name": "bad_role", "role": "nope", "max": 16},  # bad: role
        ])
        assert [p["name"] for p in normed] == ["good"]
        assert len(errors) == 2

    def test_empty_list_with_statement_is_valid(self):
        # A design with genuinely no dimensional parameters emits `[]`; the
        # prose statement lives in the ERS summary/open_items (prompt-enforced).
        normed, errors = ps.validate_parameters([])
        assert normed == []
        assert errors == []

    def test_non_list_block_is_error(self):
        normed, errors = ps.validate_parameters({"name": "x"})
        assert normed == []
        assert errors and "list" in errors[0]

    def test_role_range_supported(self):
        norm, errs = ps.normalize_parameter(
            {"name": "addr_range", "role": "range", "max": 4096})
        assert errs == []
        assert norm["role"] == "range"


# ===========================================================================
# 2. boundary_values auto-computation (2^n crossings)
# ===========================================================================
class TestBoundaryValues:
    def test_2n_crossings_up_to_max(self):
        assert ps.compute_boundary_values(640) == [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 640]

    def test_power_of_two_max_has_no_duplicate(self):
        assert ps.compute_boundary_values(256) == [
            1, 2, 4, 8, 16, 32, 64, 128, 256]

    def test_nonpositive_or_nonnumeric_is_empty(self):
        assert ps.compute_boundary_values(0) == []
        assert ps.compute_boundary_values(-5) == []
        assert ps.compute_boundary_values("frames") == []
        assert ps.compute_boundary_values(True) == []   # bool rejected

    def test_omitted_boundary_values_auto_filled(self):
        norm, _ = ps.normalize_parameter(
            {"name": "d", "role": "dimension", "max": 100})
        # every 2^n <= 100 plus 100 itself
        assert norm["boundary_values"] == [1, 2, 4, 8, 16, 32, 64, 100]

    def test_provided_boundary_values_kept_and_max_ensured(self):
        norm, _ = ps.normalize_parameter(
            {"name": "d", "role": "dimension", "max": 512,
             "boundary_values": [511, 512, 513]})
        # coerced, sorted, deduped; max 512 already a member
        assert norm["boundary_values"] == [511, 512, 513]

    def test_provided_boundary_values_append_missing_max(self):
        norm, _ = ps.normalize_parameter(
            {"name": "d", "role": "dimension", "max": 640,
             "boundary_values": [128, 512]})
        assert 640 in norm["boundary_values"]
        assert norm["boundary_values"] == [128, 512, 640]


# ===========================================================================
# 3. ERS doc validation/repair path
# ===========================================================================
class TestErsDocValidation:
    def test_normalizes_and_records_malformed_to_open_items(self):
        doc = {"ers": {"open_items": [], "parameters": [
            {"name": "keep", "role": "dimension", "max": 64},
            {"role": "dimension", "max": 8},          # malformed (no name)
        ]}}
        out, errs = ps.validate_and_normalize_ers_parameters(doc)
        assert [p["name"] for p in out["ers"]["parameters"]] == ["keep"]
        assert errs                                    # malformed reported
        assert any("parameters schema" in oi for oi in out["ers"]["open_items"])

    def test_absent_block_left_untouched(self):
        doc = {"ers": {"summary": "no params here"}}
        out, errs = ps.validate_and_normalize_ers_parameters(doc)
        assert "parameters" not in out["ers"]          # never synthesized
        assert errs == []

    def test_generate_ers_doc_normalizes_on_disk(self, tmp_path, monkeypatch):
        # Drive the real ers_doc.generate_ers_doc path: a mocked LLM writes a
        # raw (un-normalized, partly-malformed) parameters block; the validation
        # path must normalize it on disk (boundary_values filled, bad dropped).
        import asyncio
        from orchestrator.architecture.specialists import ers_doc
        from orchestrator.langchain.agents import coresmith_llm

        (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)

        class _FakeLLM:
            def __init__(self, *a, **k):
                pass

            async def call(self, system, prompt, run_name=""):
                target = tmp_path / ".coresmith" / "ers_spec.json"
                target.write_text(json.dumps({"ers": {
                    "title": "t", "summary": "s", "open_items": [],
                    "parameters": [
                        {"name": "burst_len", "role": "range", "max": 256},
                        {"name": "", "max": 8},        # malformed
                    ],
                }, "phase": "ers_complete"}))
                return "written"

        # ClaudeLLM is imported locally inside generate_ers_doc from
        # coresmith_llm -> patch it at the source module.
        monkeypatch.setattr(coresmith_llm, "ClaudeLLM", _FakeLLM)
        res = asyncio.run(ers_doc.generate_ers_doc(
            None, None, None, None, None, None, None,
            project_root=str(tmp_path)))
        disk = json.loads(
            (tmp_path / ".coresmith" / "ers_spec.json").read_text())
        params = disk["ers"]["parameters"]
        assert [p["name"] for p in params] == ["burst_len"]
        assert params[0]["boundary_values"][-1] == 256   # auto-filled
        assert res["ers"]["parameters"][0]["name"] == "burst_len"


# ===========================================================================
# 4. Consumer (a): _declared_dimensions reads the new block + fallback preserved
# ===========================================================================
class TestDeclaredDimensionsFromSchema:
    def test_reads_parameters_block_video(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_PARAMS)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        assert dims == {"frame_width": 640, "frame_height": 352}

    def test_reads_parameters_block_aes_excludes_mode(self, tmp_path):
        _write_ers(tmp_path, _AES_PARAMS)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        # dimension harvested; the mode (key_modes) is NOT a geometry axis
        assert dims == {"max_message_blocks": 1024}

    def test_fallback_preserved_no_parameters_block(self, tmp_path):
        _write_ers(tmp_path, _LEGACY_STRUCTURED)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        assert dims == {"line_len": 4096}       # generic {name,max} harvest

    def test_legacy_prose_yields_nothing(self, tmp_path):
        _write_ers(tmp_path, _LEGACY_PROSE)
        assert pipeline_graph._declared_dimensions(str(tmp_path)) == {}

    def test_schema_and_fallback_merge(self, tmp_path):
        body = dict(_VIDEO_PARAMS)
        body["constraints"] = [{"name": "coeff_depth", "max": 128}]
        _write_ers(tmp_path, body)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        assert dims == {"frame_width": 640, "frame_height": 352,
                        "coeff_depth": 128}


# ===========================================================================
# 5. maxgeo deterministic path fires END-TO-END on a schema fixture ERS
# ===========================================================================
class TestMaxgeoEndToEndFromSchema:
    def test_video_schema_missing_marker_rejected(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_PARAMS)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n# no marker\n")
        v = pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb)
        assert v is not None
        assert 640 in v["declared_dims"].values()

    def test_video_schema_marker_covers_all_passes(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_PARAMS)
        tb = _write_tb(
            tmp_path / "tb" / "t.py",
            "import cocotb\n# MAXGEO: frame_width=640 frame_height=352\n")
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None

    def test_aes_nonvideo_schema_end_to_end(self, tmp_path):
        # NON-video two-domain proof: the deterministic gate fires identically
        # off max_message_blocks with zero video vocabulary.
        _write_ers(tmp_path, _AES_PARAMS)
        tb_bad = _write_tb(tmp_path / "tb" / "b.py", "import cocotb\n")
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb_bad) is not None
        tb_ok = _write_tb(
            tmp_path / "tb" / "g.py",
            "import cocotb\n# MAXGEO: max_message_blocks=1024\n")
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb_ok) is None

    def test_legacy_prose_ers_still_noops(self, tmp_path):
        _write_ers(tmp_path, _LEGACY_PROSE)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n# no marker\n")
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None

    def test_gate_env_disable_both_branches(self, tmp_path, monkeypatch):
        _write_ers(tmp_path, _AES_PARAMS)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n")
        monkeypatch.delenv("CORESMITH_MAXGEO_GATE", raising=False)
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is not None
        monkeypatch.setenv("CORESMITH_MAXGEO_GATE", "0")
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None


# ===========================================================================
# 6. Consumer (b): the DV parameter table + (d) seeded-stimulus sizing
# ===========================================================================
class TestParameterTableConsumer:
    def test_table_rendered_verbatim(self, tmp_path):
        _write_ers(tmp_path, _AES_PARAMS)
        table = pipeline_graph._ers_parameter_table(str(tmp_path))
        assert "DIMENSIONAL PARAMETERS" in table
        assert "max_message_blocks" in table
        assert "key_modes" in table
        assert "boundary_values" in table

    def test_table_empty_when_no_params(self, tmp_path):
        _write_ers(tmp_path, _LEGACY_PROSE)
        assert pipeline_graph._ers_parameter_table(str(tmp_path)) == ""

    def test_seeded_nvectors_sees_schema_max(self, tmp_path):
        # _maybe_run_chip_equiv sizes n_vectors off max declared dim; the schema
        # now feeds the real number via _declared_dimensions.
        _write_ers(tmp_path, _AES_PARAMS)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        assert max(dims.values()) == 1024


# ===========================================================================
# 7. Consumer (c): mem-manifest strictness flips ONLY when the block exists
# ===========================================================================
class TestMemManifestStrictFlip:
    _STORAGE_SPEC = (
        "# uArch Spec: fifo_ctrl\n\n"
        "sram_budget = 16384 bits\n\n"
        "A control FIFO. (No `# MEM` manifest line -- legacy prose spec.)\n"
    )

    def test_present_block_signal(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_PARAMS)
        assert ps.ers_has_parameters_block(str(tmp_path)) is True
        assert pipeline_graph._ers_parameters_block_present(str(tmp_path)) is True

    def test_empty_block_still_present_signal(self, tmp_path):
        _write_ers(tmp_path, {"summary": "no dims", "parameters": []})
        assert ps.ers_has_parameters_block(str(tmp_path)) is True

    def test_absent_block_signal(self, tmp_path):
        _write_ers(tmp_path, _LEGACY_PROSE)
        assert ps.ers_has_parameters_block(str(tmp_path)) is False
        assert pipeline_graph._ers_parameters_block_present(str(tmp_path)) is False

    def test_legacy_run_warns_and_accepts(self, tmp_path, monkeypatch):
        # No parameters block + global default OFF -> storage-without-manifest
        # is warn-only (accept): verdict is None.
        monkeypatch.delenv("CORESMITH_MEM_MANIFEST_REQUIRED", raising=False)
        _write_ers(tmp_path, _LEGACY_PROSE)
        _write_spec(tmp_path, "fifo_ctrl", self._STORAGE_SPEC)
        v = pipeline_graph._mem_price_gate_verdict(str(tmp_path), "fifo_ctrl")
        assert v is None

    def test_new_schema_run_rejects_absent_manifest(self, tmp_path, monkeypatch):
        # Parameters block PRESENT -> strict for this run -> storage-without-
        # manifest is REJECTED (revise) even though the global default is OFF.
        monkeypatch.delenv("CORESMITH_MEM_MANIFEST_REQUIRED", raising=False)
        _write_ers(tmp_path, _VIDEO_PARAMS)
        _write_spec(tmp_path, "fifo_ctrl", self._STORAGE_SPEC)
        v = pipeline_graph._mem_price_gate_verdict(str(tmp_path), "fifo_ctrl")
        assert isinstance(v, dict)
        assert v.get("action") == "revise"
        assert "MANIFEST REQUIRED" in v.get("feedback", "")

    def test_global_optin_still_rejects_without_block(self, tmp_path, monkeypatch):
        # The global CORESMITH_MEM_MANIFEST_REQUIRED opt-in is untouched: it
        # still enforces strict even without a parameters block.
        monkeypatch.setenv("CORESMITH_MEM_MANIFEST_REQUIRED", "1")
        _write_ers(tmp_path, _LEGACY_PROSE)
        _write_spec(tmp_path, "fifo_ctrl", self._STORAGE_SPEC)
        v = pipeline_graph._mem_price_gate_verdict(str(tmp_path), "fifo_ctrl")
        assert isinstance(v, dict)
        assert v.get("action") == "revise"


# ===========================================================================
# 8. Prompt pinning (ERS mandate + DV parameter table + mem strict note)
# ===========================================================================
class TestPromptPinning:
    def _prompt(self, name: str) -> str:
        from pathlib import Path
        base = Path(pipeline_graph.__file__).resolve().parents[1]
        return (base / "langchain" / "prompts" / name).read_text()

    def test_ers_prompt_mandates_parameters_block(self):
        p = self._prompt("ers_doc.md")
        assert '"parameters"' in p
        assert "MANDATORY `parameters` block" in p
        # roles + boundary/2^n auto-fill + scope boundary crisply defined
        assert '"role"' in p
        assert "2^n" in p
        assert "pad counts" in p or "shuttle" in p   # scope exclusion

    def test_ers_prompt_allows_empty_only_with_statement(self):
        p = self._prompt("ers_doc.md")
        assert "no dimensional parameters" in p

    def test_prd_prompt_declares_parameters(self):
        p = self._prompt("prd_spec.md")
        assert '"parameters"' in p
        assert "DESIGN-PARAMETER" in p or "design-parameter" in p.lower()

    def test_dv_prompts_reference_parameters_block(self):
        for name in ("integration_testbench.md", "validation_dv.md"):
            p = self._prompt(name)
            assert "`parameters` block" in p, name
            assert "DIMENSIONAL PARAMETERS" in p, name

    def test_uarch_prompt_notes_strict_on_new_schema(self):
        p = self._prompt("uarch_spec_generator.md")
        assert "parameters" in p
        assert "STRICT on new-schema runs" in p


# ===========================================================================
# 9. [rung3r2-fixes-3] deterministic derivation fallback (unit)
# ===========================================================================
class TestDeriveParameters:
    def test_harvest_from_source_dict_tagged_derived(self):
        # A PRD-ish source carrying a {name, max} pair -> a derived dimension.
        derived = ps.derive_parameters([
            {"prd": {"constraints": [{"name": "line_len", "max": 4096}]}},
        ])
        assert [p["name"] for p in derived] == ["line_len"]
        assert derived[0]["role"] == "dimension"
        assert derived[0]["max"] == 4096
        assert derived[0]["derived"] is True
        assert derived[0]["boundary_values"][-1] == 4096   # normalized

    def test_multiple_sources_sorted_deterministic(self):
        derived = ps.derive_parameters([
            {"name": "zeta_depth", "depth": 512},
            {"name": "alpha_count", "count": 300},
            None,                                   # tolerated
        ])
        # sorted by name, deterministic
        assert [p["name"] for p in derived] == ["alpha_count", "zeta_depth"]
        assert all(p["derived"] is True for p in derived)

    def test_tiny_extents_skipped(self):
        # below the min-extent floor (8) -> not a dimension
        derived = ps.derive_parameters([{"name": "handshake", "depth": 2}])
        assert derived == []

    def test_frd_func_vector_geometry_harvested(self):
        # FRD FUNC-vector geometry (bare structured stimulus) is folded in.
        frd = (
            "## Functional Vectors\n\n"
            "### FUNC-001\n"
            "- Block: enc\n"
            "```json\n"
            '{"name": "sample_count", "max": 2048}\n'
            "```\n"
        )
        derived = ps.derive_parameters([], frd_text=frd)
        assert "sample_count" in [p["name"] for p in derived]
        got = next(p for p in derived if p["name"] == "sample_count")
        assert got["max"] == 2048 and got["derived"] is True

    def test_nothing_derivable_is_empty(self):
        derived = ps.derive_parameters([
            {"summary": "prose only, no structured dims"},
        ])
        assert derived == []

    def test_never_raises_on_garbage(self):
        assert ps.derive_parameters("not-a-list-of-dicts") == []
        assert ps.derive_parameters([{"x": object()}]) == []


# ===========================================================================
# 10. [rung3r2-fixes-3] backstop-empty present-signal (unit)
# ===========================================================================
class TestBackstopEmptyPresentSignal:
    def test_backstop_empty_reads_as_legacy(self, tmp_path):
        _write_ers(tmp_path, {
            "summary": "no dims derivable",
            "parameters": [],
            ps.BACKSTOP_MARKER_KEY: ps.BACKSTOP_EMPTY_MARKER,
        })
        # backstop-fabricated empty -> NOT a present-signal -> gates legacy
        assert ps.ers_has_parameters_block(str(tmp_path)) is False
        assert pipeline_graph._ers_parameters_block_present(str(tmp_path)) is False

    def test_genuine_empty_still_present_signal(self, tmp_path):
        # a design's OWN affirmed [] (no backstop marker) stays present
        _write_ers(tmp_path, {"summary": "no dims", "parameters": []})
        assert ps.ers_has_parameters_block(str(tmp_path)) is True


# ===========================================================================
# 11. [rung3r2-fixes-3] the backstop END-TO-END through generate_ers_doc
# ===========================================================================
def _make_scripted_llm(target, bodies):
    """Return (LLMClass, state). The class stands in for ``ClaudeLLM``; each
    ``.call`` writes the next scripted body to ``target`` (as the real generator
    LLM does) and records the prompt it was handed."""
    state = {"prompts": [], "calls": 0}

    class _LLM:
        def __init__(self, *a, **k):
            pass

        async def call(self, system, prompt, run_name=""):
            state["prompts"].append(prompt)
            body = bodies[min(state["calls"], len(bodies) - 1)]
            state["calls"] += 1
            target.write_text(json.dumps(body), encoding="utf-8")
            return "written"

    return _LLM, state


def _ers_body(**ers_extra):
    body = {"title": "t", "summary": "s", "open_items": []}
    body.update(ers_extra)
    return {"ers": body, "phase": "ers_complete"}


_ABSENT = _ers_body()                                  # NO parameters key
_PRESENT = _ers_body(parameters=[
    {"name": "burst_len", "role": "range", "max": 256}])
_AFFIRMED_EMPTY = _ers_body(
    summary="fixed-width block: no dimensional parameters", parameters=[])


class TestParametersBackstopE2E:
    def _run(self, tmp_path, monkeypatch, bodies, **kwargs):
        import asyncio
        from orchestrator.architecture.specialists import ers_doc
        from orchestrator.langchain.agents import coresmith_llm

        (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)
        target = tmp_path / ".coresmith" / "ers_spec.json"
        LLM, state = _make_scripted_llm(target, bodies)
        monkeypatch.setattr(coresmith_llm, "ClaudeLLM", LLM)
        res = asyncio.run(ers_doc.generate_ers_doc(
            kwargs.get("prd_spec"), None, kwargs.get("frd_spec"), None,
            None, None, None, project_root=str(tmp_path)))
        disk = json.loads(target.read_text(encoding="utf-8"))
        return res, disk, state

    def test_key_absent_triggers_retry_with_feedback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_ERS_PARAMS_BACKSTOP", raising=False)
        _res, _disk, state = self._run(tmp_path, monkeypatch, [_ABSENT, _ABSENT])
        assert state["calls"] == 2                      # one bounded retry
        assert "MANDATORY" in state["prompts"][1]
        assert "parameters" in state["prompts"][1]

    def test_retry_succeeds_returns_present_block(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_ERS_PARAMS_BACKSTOP", raising=False)
        res, disk, state = self._run(tmp_path, monkeypatch, [_ABSENT, _PRESENT])
        assert state["calls"] == 2
        assert [p["name"] for p in res["ers"]["parameters"]] == ["burst_len"]
        assert disk["ers"]["parameters"][0]["boundary_values"][-1] == 256
        # a successful retry is NOT a derived/empty backstop outcome
        assert ps.BACKSTOP_MARKER_KEY not in res["ers"]
        assert not any("backstop" in oi.lower()
                       for oi in res["ers"].get("open_items", []))

    def test_retry_fails_derivable_writes_tagged_block(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_ERS_PARAMS_BACKSTOP", raising=False)
        prd = {"prd": {"constraints": [{"name": "line_len", "max": 4096}]}}
        res, disk, state = self._run(
            tmp_path, monkeypatch, [_ABSENT, _ABSENT], prd_spec=prd)
        assert state["calls"] == 2
        params = res["ers"]["parameters"]
        assert [p["name"] for p in params] == ["line_len"]
        assert params[0]["derived"] is True
        assert disk["ers"]["parameters"][0]["derived"] is True
        assert any("DERIVED" in oi for oi in res["ers"]["open_items"])
        # a derived (non-empty) block IS a present-signal -> strictness active
        assert ps.ers_has_parameters_block(str(tmp_path)) is True

    def test_retry_fails_nothing_derivable_loud_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_ERS_PARAMS_BACKSTOP", raising=False)
        res, disk, state = self._run(tmp_path, monkeypatch, [_ABSENT, _ABSENT])
        assert state["calls"] == 2
        assert res["ers"]["parameters"] == []
        assert disk["ers"][ps.BACKSTOP_MARKER_KEY] == ps.BACKSTOP_EMPTY_MARKER
        assert any("legacy mode" in oi for oi in res["ers"]["open_items"])
        # loud but legacy: the fabricated empty does NOT flip strictness
        assert ps.ers_has_parameters_block(str(tmp_path)) is False
        assert pipeline_graph._ers_parameters_block_present(str(tmp_path)) is False

    def test_affirmed_empty_no_retry(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_ERS_PARAMS_BACKSTOP", raising=False)
        res, _disk, state = self._run(
            tmp_path, monkeypatch, [_AFFIRMED_EMPTY])
        assert state["calls"] == 1                       # NO retry
        assert res["ers"]["parameters"] == []
        assert ps.BACKSTOP_MARKER_KEY not in res["ers"]
        # genuine affirmed [] IS a present-signal
        assert ps.ers_has_parameters_block(str(tmp_path)) is True

    def test_env_off_preserves_today_behavior(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_ERS_PARAMS_BACKSTOP", "0")
        res, disk, state = self._run(tmp_path, monkeypatch, [_ABSENT, _PRESENT])
        assert state["calls"] == 1                       # no retry, no backstop
        assert "parameters" not in res["ers"]            # key left absent
        assert "parameters" not in disk["ers"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
