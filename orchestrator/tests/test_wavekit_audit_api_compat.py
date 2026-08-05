# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The WaveKit VCD audit must work against both WaveKit tree APIs.

WaveKit renamed its scope-tree API between 0.5.x and 0.7.x. ``wavekit`` is
unpinned in requirements.txt, and ``run_wavekit_vcd_audit``'s own fallback
pip-installs the latest into a scratch venv, so a fresh box gets the new API
regardless of what the operator pinned. The audit program used only the 0.5.x
names, so on 0.7.x it raised ``AttributeError: 'VcdReader' object has no
attribute 'top_scope_list'`` -- and because the DV gates fail CLOSED on a
non-ok audit, ``integration_dv`` and ``validation_dv`` could never pass.

These tests exec the audit program against stub readers of each shape, so both
branches are covered without needing either WaveKit version installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchestrator.langgraph.pipeline_helpers import _WAVEKIT_AUDIT_SCRIPT

_VCD = """$timescale 1ns $end
$scope module toy $end
$var wire 1 ! clk $end
$var wire 9 " s $end
$upscope $end
$enddefinitions $end
#0
0!
b0 "
#10
1!
b11 "
"""

# A stub `wavekit` module. `SHAPE` is substituted with "old" or "new".
_STUB = '''
import sys, types

SHAPE = "{shape}"


class _Sig:
    def __init__(self, full_name, width):
        self.full_name = full_name
        self.width = width


class _Scope:
    def __init__(self, sigs, subs):
        self._sigs, self._subs = sigs, subs
        if SHAPE == "old":
            self.signal_list = sigs
            self.child_scope_list = subs
        else:
            # 0.7.x: one mixed `children` tuple; only signals carry `width`.
            self.children = tuple(sigs) + tuple(subs)


class VcdReader:
    def __init__(self, path):
        self.begin_time, self.end_time = 0, 10
        inner = _Scope([_Sig("toy.sub.d[3:0]", 4)], [])
        top = _Scope([_Sig("toy.clk", 1), _Sig("toy.s[8:0]", 9)], [inner])
        if SHAPE == "old":
            self.top_scope_list = lambda: [top]
        else:
            self.top_scopes = (top,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_m = types.ModuleType("wavekit")
_m.VcdReader = VcdReader
sys.modules["wavekit"] = _m
'''


def _run_audit(tmp_path: Path, shape: str) -> dict:
    """Exec the real audit program with a stubbed wavekit of the given shape."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    vcd = tmp_path / "toy.vcd"
    vcd.write_text(_VCD)
    program = _STUB.format(shape=shape) + textwrap.dedent(_WAVEKIT_AUDIT_SCRIPT)
    proc = subprocess.run(
        [sys.executable, "-c", program, str(vcd), "clk"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"audit exited {proc.returncode}\n{proc.stderr}"
    return json.loads(proc.stdout)


class TestBothWavekitApiShapes:
    @pytest.mark.parametrize("shape", ["old", "new"])
    def test_audit_succeeds(self, tmp_path, shape):
        report = _run_audit(tmp_path, shape)
        assert report["ok"] is True

    @pytest.mark.parametrize("shape", ["old", "new"])
    def test_finds_every_signal_including_nested_scopes(self, tmp_path, shape):
        report = _run_audit(tmp_path, shape)
        assert report["signal_count"] == 3
        names = {s["name"] for s in report["sample_signals"]}
        assert names == {"toy.clk", "toy.s[8:0]", "toy.sub.d[3:0]"}

    @pytest.mark.parametrize("shape", ["old", "new"])
    def test_widths_preserved(self, tmp_path, shape):
        widths = {s["name"]: s["width"] for s in _run_audit(tmp_path, shape)["sample_signals"]}
        assert widths == {"toy.clk": 1, "toy.s[8:0]": 9, "toy.sub.d[3:0]": 4}

    @pytest.mark.parametrize("shape", ["old", "new"])
    def test_clock_detected_despite_range_suffix(self, tmp_path, shape):
        assert _run_audit(tmp_path, shape)["clock_candidates"] == ["toy.clk"]

    def test_both_shapes_agree(self, tmp_path):
        """The shim must not change what the audit reports, only how it walks."""
        old = _run_audit(tmp_path / "a", "old")
        new = _run_audit(tmp_path / "b", "new")
        for key in ("ok", "signal_count", "clock_candidates", "sample_signals"):
            assert old[key] == new[key], key


class TestAgainstInstalledWavekit:
    def test_real_wavekit_audit_passes(self, tmp_path):
        """End-to-end against whatever WaveKit is actually installed."""
        pytest.importorskip("wavekit")
        from orchestrator.langgraph.pipeline_helpers import run_wavekit_vcd_audit

        vcd = tmp_path / "toy.vcd"
        vcd.write_text(_VCD)
        report = run_wavekit_vcd_audit(vcd, tmp_path / "audit.json")
        # A WaveKit that cannot be set up is reported as skipped-but-ok; that is
        # a deliberate separate path and not what this test is about.
        if report.get("skipped"):
            pytest.skip(f"WaveKit unavailable: {report.get('reason')}")
        assert report["ok"] is True, report
        assert report["signal_count"] == 2
        assert report["clock_candidates"] == ["toy.clk"]
