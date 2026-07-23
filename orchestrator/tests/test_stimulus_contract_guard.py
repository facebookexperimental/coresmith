"""Tests for the generic stimulus<->contract consistency guard."""
import json

import pytest

from orchestrator.architecture import stimulus_contract_guard as scg

_ENV = [
    "CORESMITH_STIMULUS_CONTRACT_GUARD",
    "CORESMITH_MODEL_STIMULUS",
    "CORESMITH_SOURCE_ROOT",
    "CORESMITH_REFERENCE_ENTRY",
]


def _clear(monkeypatch):
    for e in _ENV:
        monkeypatch.delenv(e, raising=False)


def _write_block_diagram(root, blocks, connections):
    cs = root / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    (cs / "block_diagram.json").write_text(
        json.dumps({"blocks": blocks, "connections": connections})
    )


# Codec-like: frame_ctrl exposes a pixel stream + a qp/lifecycle config input,
# but NO height/width boundary input (geometry was meant to be hardcoded).
_CODEC_BLOCKS = [
    {"name": "frame_ctrl", "interfaces": {
        "cfg_lifecycle_in": {"type": "pins", "signals": {
            "frame_start_i": 1, "frame_end_flush_i": 1, "cfg_qp_i": 6}},
        "s_axis_pixel_in": {"type": "axi_stream", "tdata_width": 8},
        "config_axis_out": {"type": "axi_stream"},
    }},
    {"name": "stripe", "interfaces": {
        "config_axis_in": {"type": "axi_stream"},
        "macroblock_axis_out": {"type": "axi_stream"},
    }},
]
_CODEC_CONNS = [{"from": "frame_ctrl", "to": "stripe", "interface": "config_axis"}]


# ---- pure matching units (no env / no I/O) --------------------------------

class TestTokenizeAndMatch:
    def test_tokenize_camel_and_snake(self):
        assert scg._tokenize("cfg_qp_i") == {"cfg", "qp", "i"}
        assert scg._tokenize("frameHeightIn") == {"frame", "height", "in"}

    def test_short_field_no_substring_false_positive(self):
        # 'h' must NOT match the 'h' inside 'flush'.
        toks = {"frame", "end", "flush", "qp"}
        assert scg._field_covered("h", toks) is False
        assert scg._field_covered("w", toks) is False

    def test_qp_exact_token_match(self):
        assert scg._field_covered("qp", {"cfg", "qp", "i"}) is True

    def test_height_alias_match(self):
        assert scg._field_covered("h", {"frame", "height", "in"}) is True
        assert scg._field_covered("w", {"img", "width"}) is True

    def test_payload_detection(self):
        assert scg._is_payload([[1, 2], [3, 4]]) is True       # nested
        assert scg._is_payload(list(range(50))) is True        # long vector
        assert scg._is_payload(36) is False                    # scalar
        assert scg._is_payload([1, 2, 3]) is False             # short vector

    def test_config_fields_drops_payload(self):
        stim = {"pixels": [[1, 2], [3, 4]], "qp": 36, "H": 16, "W": 16}
        assert scg._stimulus_config_fields(stim) == {"qp": 36, "H": 16, "W": 16}


class TestExternalInputTokens:
    def test_boundary_inputs_only(self, tmp_path):
        _write_block_diagram(tmp_path, _CODEC_BLOCKS, _CODEC_CONNS)
        toks = scg.external_input_tokens(str(tmp_path))
        assert toks is not None
        # qp + pixel boundary inputs present; the internally-driven
        # config_axis_in (a connection destination) is excluded.
        assert "qp" in toks
        assert "pixel" in toks
        # no height/width concept anywhere.
        assert not ({"height", "width", "rows", "cols"} & toks)

    def test_missing_block_diagram_returns_none(self, tmp_path):
        assert scg.external_input_tokens(str(tmp_path)) is None

    def test_axi_string_schema_with_payload_fields(self, tmp_path):
        # The other observed schema: AXI s_axis/m_axis naming, `signals` as a
        # STRING, and real field names only inside a `payload` string as
        # name[width]. The geometry-parametric design carries frame_h/frame_w.
        blocks = [
            {"name": "frame_lifecycle_ctrl", "interfaces": {
                "s_axis_frame": {
                    "signals": "tdata[41:0], tvalid, tready, tlast",
                    "payload": ("payload_width = frame_h[10] + frame_w[10] + "
                                "frame_qp[6] + frame_id[16] = 42 bits")},
                "s_axis_pix": {"signals": "tdata[7:0], tvalid, tready"},
                "m_axis_frame_cfg": {"signals": "tdata[41:0], tvalid"},
            }},
            {"name": "stripe", "interfaces": {
                "s_axis_frame_cfg": {"signals": "tdata[41:0], tvalid"},
                "m_axis_mb": {"signals": "tdata[7:0], tvalid"},
            }},
        ]
        conns = [{"from": "frame_lifecycle_ctrl", "to": "stripe",
                  "interface": "m_axis_frame_cfg -> s_axis_frame_cfg"}]
        _write_block_diagram(tmp_path, blocks, conns)
        toks = scg.external_input_tokens(str(tmp_path))
        assert toks is not None
        # frame_h/frame_w/frame_qp extracted from the boundary s_axis_frame.
        assert {"h", "w", "qp", "frame"} <= toks
        # geometry IS carried -> H/W/qp all covered (no false flags).
        assert scg._field_covered("h", toks)
        assert scg._field_covered("w", toks)
        assert scg._field_covered("qp", toks)
        # 'payload_width' prose must NOT leak a spurious 'width' token.
        assert "width" not in toks


# ---- end-to-end guard -----------------------------------------------------

def _setup_codec(tmp_path, monkeypatch, ref_body):
    _clear(monkeypatch)
    _write_block_diagram(tmp_path, _CODEC_BLOCKS, _CODEC_CONNS)
    ref = tmp_path / "ref.py"
    ref.write_text(ref_body)
    stim = tmp_path / "stim.py"
    stim.write_text(
        "stimulus = {'pixels': [[1, 2], [3, 4]], 'qp': 36, 'H': 16, 'W': 16}\n"
    )
    monkeypatch.setenv("CORESMITH_SOURCE_ROOT", str(ref))
    monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "encode")
    monkeypatch.setenv("CORESMITH_MODEL_STIMULUS", str(stim))


class TestGuardEndToEnd:
    def test_geometry_gap_flags_h_and_w_only(self, tmp_path, monkeypatch):
        _setup_codec(tmp_path, monkeypatch,
                     "def encode(pixels=None, qp=None, H=None, W=None, **kw):\n    return [1, 2, 3, 4, 5]\n")
        v = scg.run_stimulus_contract_guard(str(tmp_path))
        cov = {x["field"] for x in v
               if x["type"] == "stimulus_field_no_external_input"}
        assert cov == {"H", "W"}, v
        # qp + the pixels payload are NOT flagged.
        assert "qp" not in cov
        # No oracle violation (reference accepted + returned non-degenerate).
        assert not [x for x in v if x["severity"] == "error"]

    def test_oracle_reject_is_error(self, tmp_path, monkeypatch):
        _setup_codec(
            tmp_path, monkeypatch,
            "def encode(pixels, qp, H, W):\n"
            "    raise ValueError('bad frame')\n",
        )
        v = scg.run_stimulus_contract_guard(str(tmp_path))
        errs = [x for x in v if x["type"] == "oracle_reference_rejects_stimulus"]
        assert len(errs) == 1
        assert errs[0]["severity"] == "error"

    def test_all_config_covered_clean(self, tmp_path, monkeypatch):
        # Same stimulus, but the design DOES expose height/width boundary inputs.
        _clear(monkeypatch)
        blocks = [{"name": "frame_ctrl", "interfaces": {
            "cfg_in": {"type": "pins", "signals": {
                "cfg_qp_i": 6, "frame_height_i": 10, "frame_width_i": 10}},
            "s_axis_pixel_in": {"type": "axi_stream"},
        }}]
        _write_block_diagram(tmp_path, blocks, [])
        ref = tmp_path / "ref.py"
        ref.write_text("def encode(pixels=None, qp=None, H=None, W=None, **kw):\n    return [1, 2, 3, 4]\n")
        stim = tmp_path / "stim.py"
        stim.write_text(
            "stimulus = {'pixels': [[1,2],[3,4]], 'qp': 36, 'H': 16, 'W': 16}\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT", str(ref))
        monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "encode")
        monkeypatch.setenv("CORESMITH_MODEL_STIMULUS", str(stim))
        v = scg.run_stimulus_contract_guard(str(tmp_path))
        assert not [x for x in v
                    if x["type"] == "stimulus_field_no_external_input"], v

    def test_non_dict_stimulus_no_coverage(self, tmp_path, monkeypatch):
        _clear(monkeypatch)
        _write_block_diagram(tmp_path, _CODEC_BLOCKS, _CODEC_CONNS)
        ref = tmp_path / "ref.py"
        ref.write_text("def encode(x):\n    return [v+1 for v in x]\n")
        stim = tmp_path / "stim.py"
        stim.write_text("stimulus = [1, 2, 3, 4]\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT", str(ref))
        monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "encode")
        monkeypatch.setenv("CORESMITH_MODEL_STIMULUS", str(stim))
        v = scg.run_stimulus_contract_guard(str(tmp_path))
        assert not [x for x in v
                    if x["type"] == "stimulus_field_no_external_input"]

    def test_guard_off_is_noop(self, tmp_path, monkeypatch):
        _setup_codec(tmp_path, monkeypatch,
                     "def encode(pixels=None, qp=None, H=None, W=None, **kw):\n    return [1, 2, 3]\n")
        monkeypatch.setenv("CORESMITH_STIMULUS_CONTRACT_GUARD", "off")
        assert scg.run_stimulus_contract_guard(str(tmp_path)) == []

    def test_no_stimulus_is_noop(self, tmp_path, monkeypatch):
        _clear(monkeypatch)
        _write_block_diagram(tmp_path, _CODEC_BLOCKS, _CODEC_CONNS)
        assert scg.run_stimulus_contract_guard(str(tmp_path)) == []


class TestGuardFlags:
    def test_enabled_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_STIMULUS_CONTRACT_GUARD", raising=False)
        assert scg.guard_enabled() is True
        assert scg.guard_strict() is False

    def test_off_and_strict(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_STIMULUS_CONTRACT_GUARD", "off")
        assert scg.guard_enabled() is False
        monkeypatch.setenv("CORESMITH_STIMULUS_CONTRACT_GUARD", "strict")
        assert scg.guard_strict() is True
        assert scg.guard_enabled() is True
