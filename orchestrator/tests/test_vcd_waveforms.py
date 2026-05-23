# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for VCD parsing, WaveDrom conversion, and dashboard HTML injection.

Covers:
- VCD header parsing (variables, scopes)
- VCD value-change parsing (scalar + vector)
- Clock period auto-detection
- Full VCD → WaveDrom JSON conversion
- 1-bit signal compression
- Multi-bit bus data label extraction
- Edge cases: empty VCD, no clock, oversized files
- Deterministic HTML injection (inject_vcd_waveforms)
"""

import json
import textwrap
from pathlib import Path

import pytest

from orchestrator.architecture.specialists.chip_finish_dashboard import (
    _parse_vcd_header,
    _parse_vcd_values,
    _detect_clock_period,
    _vcd_to_wavedrom,
    _collect_vcd_waveforms,
    inject_vcd_waveforms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_VCD = textwrap.dedent("""\
    $date Mon Jan 1 00:00:00 2024 $end
    $version Verilator 5.0 $end
    $timescale 1ns $end
    $scope module top $end
    $var wire 1 ! clk $end
    $var wire 1 " rst_n $end
    $var wire 1 # valid $end
    $var wire 8 $ data [7:0] $end
    $upscope $end
    $enddefinitions $end
    #0
    0!
    0"
    0#
    b00000000 $
    #5
    1!
    #10
    0!
    1"
    #15
    1!
    #20
    0!
    1#
    b00000001 $
    #25
    1!
    #30
    0!
    b00000010 $
    #35
    1!
    #40
    0!
    0#
    b00000000 $
    #45
    1!
    #50
    0!
""")

MULTI_SCOPE_VCD = textwrap.dedent("""\
    $timescale 1ps $end
    $scope module chip $end
    $var wire 1 A clk $end
    $var wire 1 B rst $end
    $scope module alu $end
    $var wire 8 C op [7:0] $end
    $var wire 16 D result [15:0] $end
    $upscope $end
    $scope module mem $end
    $var wire 1 E we $end
    $upscope $end
    $upscope $end
    $enddefinitions $end
    #0
    0A
    1B
    0E
    b00000000 C
    b0000000000000000 D
    #5
    1A
    #10
    0A
    0B
    b00000001 C
    #15
    1A
    #20
    0A
    b0000000000001010 D
    1E
    #25
    1A
    #30
    0A
    0E
""")


@pytest.fixture
def minimal_vcd_file(tmp_path):
    p = tmp_path / "dump.vcd"
    p.write_text(MINIMAL_VCD)
    return p


@pytest.fixture
def multi_scope_vcd_file(tmp_path):
    p = tmp_path / "dump.vcd"
    p.write_text(MULTI_SCOPE_VCD)
    return p


# ---------------------------------------------------------------------------
# _parse_vcd_header
# ---------------------------------------------------------------------------


class TestParseVcdHeader:
    def test_parses_variables(self):
        lines = MINIMAL_VCD.splitlines()
        variables, body_start = _parse_vcd_header(lines)

        assert len(variables) == 4
        assert "!" in variables
        assert variables["!"]["name"] == "clk"
        assert variables["!"]["width"] == 1
        assert variables["$"]["name"] == "data"
        assert variables["$"]["width"] == 8

    def test_scope_assignment(self):
        lines = MINIMAL_VCD.splitlines()
        variables, _ = _parse_vcd_header(lines)

        for var in variables.values():
            assert var["scope"] == "top"

    def test_multi_scope(self):
        lines = MULTI_SCOPE_VCD.splitlines()
        variables, _ = _parse_vcd_header(lines)

        assert variables["A"]["scope"] == "chip"
        assert variables["C"]["scope"] == "chip.alu"
        assert variables["D"]["scope"] == "chip.alu"
        assert variables["E"]["scope"] == "chip.mem"

    def test_body_start_after_enddefinitions(self):
        lines = MINIMAL_VCD.splitlines()
        _, body_start = _parse_vcd_header(lines)

        remaining = lines[body_start:]
        time_lines = [line for line in remaining if line.strip().startswith("#")]
        assert len(time_lines) > 0

    def test_empty_vcd(self):
        variables, body_start = _parse_vcd_header([])
        assert variables == {}

    def test_no_variables(self):
        lines = ["$enddefinitions $end", "#0", "1!"]
        variables, _ = _parse_vcd_header(lines)
        assert variables == {}


# ---------------------------------------------------------------------------
# _parse_vcd_values
# ---------------------------------------------------------------------------


class TestParseVcdValues:
    def test_scalar_values(self):
        lines = MINIMAL_VCD.splitlines()
        variables, body_start = _parse_vcd_header(lines)
        changes = _parse_vcd_values(lines, body_start, variables)

        clk_changes = changes["!"]
        assert len(clk_changes) > 0
        assert clk_changes[0] == (0, "0")
        assert clk_changes[1] == (5, "1")

    def test_vector_values(self):
        lines = MINIMAL_VCD.splitlines()
        variables, body_start = _parse_vcd_header(lines)
        changes = _parse_vcd_values(lines, body_start, variables)

        data_changes = changes["$"]
        assert data_changes[0] == (0, "00000000")
        assert data_changes[1] == (20, "00000001")
        assert data_changes[2] == (30, "00000010")

    def test_time_ordering(self):
        lines = MINIMAL_VCD.splitlines()
        variables, body_start = _parse_vcd_header(lines)
        changes = _parse_vcd_values(lines, body_start, variables)

        for ident, trans in changes.items():
            times = [t for t, _ in trans]
            assert times == sorted(times), f"Signal {ident} not time-ordered"

    def test_unknown_identifiers_ignored(self):
        lines = ["#0", "1Z", "b1111 Q", "#10", "0Z"]
        changes = _parse_vcd_values(lines, 0, {"!": {"name": "clk", "width": 1, "scope": "top"}})
        assert all(len(v) == 0 for v in changes.values())


# ---------------------------------------------------------------------------
# _detect_clock_period
# ---------------------------------------------------------------------------


class TestDetectClockPeriod:
    def test_finds_clock(self):
        lines = MINIMAL_VCD.splitlines()
        variables, body_start = _parse_vcd_header(lines)
        changes = _parse_vcd_values(lines, body_start, variables)

        clk_id, period = _detect_clock_period(changes, variables)
        assert clk_id == "!"
        assert period == 10

    def test_no_clock_signal(self):
        variables = {"$": {"name": "data", "width": 8, "scope": "top"}}
        changes = {"$": [(0, "00000000"), (10, "00000001")]}
        clk_id, period = _detect_clock_period(changes, variables)
        assert clk_id == ""
        assert period == 0

    def test_too_few_transitions(self):
        variables = {"!": {"name": "clk", "width": 1, "scope": "top"}}
        changes = {"!": [(0, "0"), (5, "1")]}
        clk_id, period = _detect_clock_period(changes, variables)
        assert period == 0


# ---------------------------------------------------------------------------
# _vcd_to_wavedrom
# ---------------------------------------------------------------------------


class TestVcdToWavedrom:
    def test_basic_conversion(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)

        assert result is not None
        assert "signal" in result
        assert "config" in result
        assert len(result["signal"]) > 0

    def test_clock_signal_first(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)
        assert result["signal"][0]["name"] == "clk"
        assert result["signal"][0]["wave"].startswith("p")

    def test_1bit_compression(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)

        rst_sig = next(s for s in result["signal"] if s["name"] == "rst_n")
        wave = rst_sig["wave"]
        assert wave.startswith("01")
        assert ".." in wave, "Repeated values should compress to dots"
        assert wave.count("1") == 1, "Only one '1' transition, rest should be dots"

    def test_bus_has_data_labels(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)

        data_sig = next(s for s in result["signal"] if s["name"] == "data")
        assert "data" in data_sig
        assert isinstance(data_sig["data"], list)
        assert len(data_sig["data"]) > 0
        assert all(d.startswith("0x") for d in data_sig["data"])

    def test_bus_hex_values_correct(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)

        data_sig = next(s for s in result["signal"] if s["name"] == "data")
        assert "0x0" in data_sig["data"]
        assert "0x1" in data_sig["data"]
        assert "0x2" in data_sig["data"]

    def test_hscale_set(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)
        assert result["config"]["hscale"] == 2

    def test_nonexistent_file(self, tmp_path):
        result = _vcd_to_wavedrom(tmp_path / "nope.vcd")
        assert result is None

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.vcd"
        p.write_text("")
        result = _vcd_to_wavedrom(p)
        assert result is None

    def test_no_clock_returns_none(self, tmp_path):
        vcd = textwrap.dedent("""\
            $timescale 1ns $end
            $scope module top $end
            $var wire 8 $ data [7:0] $end
            $upscope $end
            $enddefinitions $end
            #0
            b00000000 $
            #10
            b00000001 $
        """)
        p = tmp_path / "noclock.vcd"
        p.write_text(vcd)
        result = _vcd_to_wavedrom(p)
        assert result is None

    def test_oversized_file_skipped(self, tmp_path):
        p = tmp_path / "huge.vcd"
        p.write_text("x" * (51 * 1024 * 1024))
        result = _vcd_to_wavedrom(p)
        assert result is None

    def test_multi_scope_signals(self, multi_scope_vcd_file):
        result = _vcd_to_wavedrom(multi_scope_vcd_file)
        assert result is not None
        names = [s["name"] for s in result["signal"]]
        assert "clk" in names

    def test_all_wave_strings_same_length(self, minimal_vcd_file):
        result = _vcd_to_wavedrom(minimal_vcd_file)
        waves = [s["wave"] for s in result["signal"]]
        lengths = set(len(w) for w in waves)
        assert len(lengths) == 1, f"Wave lengths differ: {lengths}"

    def test_valid_wavedrom_json(self, minimal_vcd_file):
        """Output should be valid JSON that WaveDrom can consume."""
        result = _vcd_to_wavedrom(minimal_vcd_file)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["signal"] == result["signal"]


# ---------------------------------------------------------------------------
# _collect_vcd_waveforms
# ---------------------------------------------------------------------------


class TestCollectVcdWaveforms:
    def test_finds_block_vcds(self, tmp_path):
        sim_dir = tmp_path / "sim_build" / "my_block"
        sim_dir.mkdir(parents=True)
        (sim_dir / "dump.vcd").write_text(MINIMAL_VCD)

        blocks = [{"name": "my_block", "success": True}]
        result = _collect_vcd_waveforms(tmp_path, blocks)

        assert len(result) == 1
        assert result[0]["block_name"] == "my_block"
        assert result[0]["source"] == "block"
        assert result[0]["passed"] is True
        assert result[0]["wavedrom"] is not None

    def test_finds_integration_vcd(self, tmp_path):
        sim_dir = tmp_path / "sim_build" / "integration"
        sim_dir.mkdir(parents=True)
        (sim_dir / "dump.vcd").write_text(MINIMAL_VCD)

        result = _collect_vcd_waveforms(tmp_path, [])
        assert len(result) == 1
        assert result[0]["block_name"] == "integration"
        assert result[0]["source"] == "integration"

    def test_missing_vcd_skipped(self, tmp_path):
        blocks = [{"name": "ghost_block", "success": True}]
        result = _collect_vcd_waveforms(tmp_path, blocks)
        assert len(result) == 0

    def test_failed_block_still_collected(self, tmp_path):
        sim_dir = tmp_path / "sim_build" / "buggy"
        sim_dir.mkdir(parents=True)
        (sim_dir / "dump.vcd").write_text(MINIMAL_VCD)

        blocks = [{"name": "buggy", "success": False}]
        result = _collect_vcd_waveforms(tmp_path, blocks)
        assert len(result) == 1
        assert result[0]["passed"] is False

    def test_multiple_blocks(self, tmp_path):
        for name in ("alu", "mem", "ctrl"):
            d = tmp_path / "sim_build" / name
            d.mkdir(parents=True)
            (d / "dump.vcd").write_text(MINIMAL_VCD)

        blocks = [
            {"name": "alu", "success": True},
            {"name": "mem", "success": True},
            {"name": "ctrl", "success": False},
        ]
        result = _collect_vcd_waveforms(tmp_path, blocks)
        assert len(result) == 3
        names = [r["block_name"] for r in result]
        assert names == ["alu", "mem", "ctrl"]


# ---------------------------------------------------------------------------
# inject_vcd_waveforms
# ---------------------------------------------------------------------------

MINIMAL_HTML = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Test Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    </head>
    <body>
    <main>
    <section id="summary"><h2>Summary</h2></section>
    <section id="waveform"><h2>Expected Waveform</h2><svg>...</svg></section>
    <section id="rtl"><h2>RTL</h2></section>
    </main>
    </body>
    </html>
""")


def _make_waveform_entry(name="test_block", passed=True):
    return {
        "block_name": name,
        "source": "block",
        "passed": passed,
        "wavedrom": {
            "signal": [
                {"name": "clk", "wave": "p..."},
                {"name": "data", "wave": "=.=.", "data": ["0x0", "0x1"]},
            ],
            "config": {"hscale": 2},
        },
    }


class TestInjectVcdWaveforms:
    def test_injects_wavedrom_cdn(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert "wavedrom.com/skins/default.js" in result
        assert "wavedrom.com/wavedrom.min.js" in result

    def test_cdn_in_head(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        head_end = result.index("</head>")
        assert "wavedrom.min.js" in result[:head_end]

    def test_no_duplicate_cdn(self):
        html_with_cdn = MINIMAL_HTML.replace(
            "</head>",
            '<script src="https://wavedrom.com/wavedrom.min.js"></script>\n</head>',
        )
        result = inject_vcd_waveforms(html_with_cdn, [_make_waveform_entry()])
        assert result.count("wavedrom.min.js") == 1

    def test_replaces_existing_waveform_section(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert "Expected Waveform" not in result
        assert "Simulation Waveforms" in result
        assert 'id="waveform"' in result

    def test_preserves_other_sections(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert 'id="summary"' in result
        assert 'id="rtl"' in result

    def test_includes_block_name(self):
        result = inject_vcd_waveforms(
            MINIMAL_HTML, [_make_waveform_entry("my_adder")]
        )
        assert "my_adder" in result

    def test_pass_badge(self):
        result = inject_vcd_waveforms(
            MINIMAL_HTML, [_make_waveform_entry(passed=True)]
        )
        assert "PASS" in result

    def test_fail_badge(self):
        result = inject_vcd_waveforms(
            MINIMAL_HTML, [_make_waveform_entry(passed=False)]
        )
        assert "FAIL" in result

    def test_wavedrom_json_embedded(self):
        entry = _make_waveform_entry()
        result = inject_vcd_waveforms(MINIMAL_HTML, [entry])
        assert '"wave": "p..."' in result
        assert 'type="WaveDrom"' in result

    def test_lazy_rendering(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert "_vcdToggle" in result
        assert "data-vcd-card" in result

    def test_cards_collapsed_by_default(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert 'display:none' in result

    def test_first_card_auto_expands_on_load(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert "DOMContentLoaded" in result
        assert "_vcdToggle" in result

    def test_scrollable_container(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert "overflow-x:auto" in result

    def test_multiple_blocks(self):
        entries = [
            _make_waveform_entry("block_a", True),
            _make_waveform_entry("block_b", False),
        ]
        result = inject_vcd_waveforms(MINIMAL_HTML, entries)
        assert "block_a" in result
        assert "block_b" in result
        assert result.count('<script type="WaveDrom">') == 2
        assert result.count('onclick="_vcdToggle(this)"') == 2

    def test_many_blocks_all_present(self):
        entries = [_make_waveform_entry(f"block_{i}") for i in range(20)]
        result = inject_vcd_waveforms(MINIMAL_HTML, entries)
        assert result.count('onclick="_vcdToggle(this)"') == 20
        for i in range(20):
            assert f"block_{i}" in result

    def test_empty_waveforms_no_change(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [])
        assert result == MINIMAL_HTML

    def test_no_waveform_section_appends_before_main(self):
        html_no_waveform = MINIMAL_HTML.replace(
            '<section id="waveform"><h2>Expected Waveform</h2><svg>...</svg></section>\n',
            "",
        )
        result = inject_vcd_waveforms(html_no_waveform, [_make_waveform_entry()])
        assert 'id="waveform"' in result
        assert "_vcdToggle" in result

    def test_signal_and_cycle_counts_in_header(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert "2 signals" in result
        assert "4 cycles" in result

    def test_output_is_valid_html(self):
        result = inject_vcd_waveforms(MINIMAL_HTML, [_make_waveform_entry()])
        assert result.count("<html") == 1
        assert result.count("</html>") == 1
        assert result.count("<head>") == 1
        assert result.count("</head>") == 1
        assert result.count("</body>") == 1


# ---------------------------------------------------------------------------
# Real VCD file test (if exp5 data is available)
# ---------------------------------------------------------------------------


class TestRealVcd:
    """Tests against actual simulation output from exp5 (skipped if missing)."""

    VCD_PATH = Path.home() / "coresmith_experiments" / "exp5" / "sim_build" / "adder_8bit" / "dump.vcd"

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not self.VCD_PATH.exists():
            pytest.skip("exp5 VCD not available")

    def test_parses_real_vcd(self):
        result = _vcd_to_wavedrom(self.VCD_PATH)
        assert result is not None
        assert len(result["signal"]) >= 5

    def test_real_signals_present(self):
        result = _vcd_to_wavedrom(self.VCD_PATH)
        names = [s["name"] for s in result["signal"]]
        assert "clk" in names
        assert "sum" in names or "a" in names

    def test_real_cycle_count(self):
        result = _vcd_to_wavedrom(self.VCD_PATH)
        wave_len = len(result["signal"][0]["wave"])
        assert wave_len > 100, f"Expected >100 cycles from full test suite, got {wave_len}"

    def test_real_vcd_produces_valid_json(self):
        result = _vcd_to_wavedrom(self.VCD_PATH)
        serialized = json.dumps(result)
        assert len(serialized) > 100
        parsed = json.loads(serialized)
        assert parsed == result
