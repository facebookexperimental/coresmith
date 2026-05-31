"""WaveKit VCD audit graceful-degradation.

The WaveKit audit is supplementary on top of the cocotb regression result. If
WaveKit can't be set up (no prebuilt wheel for the arch + missing build deps),
the audit must SKIP gracefully rather than report ok=False -- otherwise a pure
infrastructure problem masquerades as an integration_dv failure (DV_PROCESS_ERROR).
"""
import json
import subprocess

from orchestrator.langgraph import pipeline_helpers


def test_wavekit_setup_failure_skips_not_fails(tmp_path, monkeypatch):
    vcd = tmp_path / "dump.vcd"
    vcd.write_text("$timescale 1ns $end\n")  # non-empty so we pass the size guard
    audit = tmp_path / "wavekit_audit.json"

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0] if args else "cmd",
                                            output="", stderr="Python.h: No such file")

    # Force WaveKit setup to fail (no wheel, can't compile pylibfst).
    monkeypatch.setattr(pipeline_helpers.subprocess, "run", boom)

    result = pipeline_helpers.run_wavekit_vcd_audit(vcd, audit)

    # Non-fatal: ok stays True (so callers don't fail DV), but clearly skipped.
    assert result["ok"] is True
    assert result["skipped"] is True
    assert "WaveKit unavailable" in result["reason"]
    # audit file is still written for forensics
    assert json.loads(audit.read_text())["skipped"] is True


def test_wavekit_missing_vcd_is_not_skipped(tmp_path):
    # A genuinely missing/empty VCD is a real error (ok=False), not a skip.
    audit = tmp_path / "wavekit_audit.json"
    result = pipeline_helpers.run_wavekit_vcd_audit(tmp_path / "nope.vcd", audit)
    assert result["ok"] is False
    assert not result.get("skipped")
