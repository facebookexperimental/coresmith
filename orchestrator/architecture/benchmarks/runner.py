# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark synthesis runner.

Generates parameterized micro-benchmark RTL from Jinja2 templates,
synthesizes with Yosys, optionally runs OpenSTA for timing, and caches
results in SQLite.

Reuses the Yosys script generation pattern from
orchestrator/temporal/activities/eda_activities.py but with shorter
timeouts suited for small benchmark circuits.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from orchestrator.pdk.pdk_config import PDKConfig

from .cache import BenchmarkCache

# Timeout for benchmark synthesis (much shorter than production 1-hour timeout)
BENCHMARK_SYNTH_TIMEOUT_S = 60
BENCHMARK_STA_TIMEOUT_S = 30

# Directory for generated benchmark RTL and synthesis output
BENCHMARK_OUTPUT_DIR = ".coresmith/benchmarks"

# Templates directory (relative to this file)
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _render_template(component: str, params: dict) -> str:
    """Render a Jinja2 Verilog template with the given parameters."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(f"{component}.v.j2")
    return template.render(**params)


def _module_name(component: str) -> str:
    """Get the Verilog module name for a benchmark component."""
    return f"benchmark_{component}"


def _params_hash(component: str, params: dict) -> str:
    """Short hash of component + params for file naming."""
    import json

    canonical = f"{component}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _parse_gate_count(stdout: str) -> int:
    """Parse gate count from Yosys stat output."""
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("Number of cells:"):
            try:
                return int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        if line.endswith(" cells") and not line.startswith("Chip"):
            try:
                return int(line.replace("cells", "").strip())
            except ValueError:
                pass
    return 0


def _parse_sta_slack(report: str) -> float | None:
    """Parse worst slack from OpenSTA output."""
    for line in report.split("\n"):
        if "slack" in line.lower():
            parts = line.split()
            for part in parts:
                try:
                    return float(part)
                except ValueError:
                    continue
    return None


async def run_benchmark(
    component: str,
    params: dict,
    pdk_config: PDKConfig,
    target_clock_mhz: float = 50.0,
    project_root: str = ".",
    skip_sta: bool = False,
) -> dict[str, Any]:
    """Synthesize a micro-benchmark and return empirical results.

    Args:
        component: Benchmark type ("multiplier", "fifo", "sram_array",
                   "fft_butterfly", "counter").
        params: Template parameters (e.g. {"width": 16}).
        pdk_config: PDK configuration for synthesis targeting.
        target_clock_mhz: Target clock frequency.
        project_root: Project root for output paths.
        skip_sta: If True, skip OpenSTA timing analysis.

    Returns:
        Dict with keys: component, params, gate_count, area_um2,
        max_clock_mhz, worst_slack_ns, synthesis_time_s, cached.
    """
    # Check cache first
    cache_db = os.path.join(project_root, ".coresmith", "benchmark_cache.db")
    cache = BenchmarkCache(cache_db)

    cached = cache.get(component, params, pdk_config.name, target_clock_mhz)
    if cached is not None:
        cache.close()
        return cached

    # Set up output directory
    ph = _params_hash(component, params)
    out_dir = Path(project_root) / BENCHMARK_OUTPUT_DIR / f"{component}_{ph}"
    out_dir.mkdir(parents=True, exist_ok=True)

    module_name = _module_name(component)
    rtl_path = out_dir / f"{module_name}.v"
    netlist_path = out_dir / f"{module_name}_synth.v"
    report_path = out_dir / f"{module_name}_report.txt"
    script_path = out_dir / f"{module_name}_synth.ys"

    # Render template
    try:
        verilog = _render_template(component, params)
    except Exception as e:
        cache.close()
        return {
            "component": component,
            "params": params,
            "gate_count": 0,
            "area_um2": 0,
            "max_clock_mhz": 0,
            "worst_slack_ns": None,
            "synthesis_time_s": 0,
            "cached": False,
            "error": f"Template rendering failed: {e}",
        }

    rtl_path.write_text(verilog)

    # Build Yosys synthesis script
    liberty = pdk_config.liberty_path()
    clock_period_ns = 1000.0 / target_clock_mhz

    if Path(liberty).exists():
        yosys_script = f"""\
# Benchmark synthesis: {component} {params}
read_verilog {rtl_path}
hierarchy -check -top {module_name}
synth -top {module_name}
dfflibmap -liberty {liberty}
abc -liberty {liberty}
clean
stat -liberty {liberty}
tee -o {report_path} stat
write_verilog -noattr {netlist_path}
"""
    else:
        yosys_script = f"""\
# Benchmark synthesis (no PDK): {component} {params}
read_verilog {rtl_path}
hierarchy -check -top {module_name}
proc; opt; fsm; opt; memory; opt
synth -top {module_name}
clean
tee -o {report_path} stat
write_verilog -noattr {netlist_path}
"""

    script_path.write_text(yosys_script)

    # Run Yosys
    start_time = time.monotonic()
    try:
        result = subprocess.run(
            ["yosys", "-s", str(script_path)],
            capture_output=True,
            text=True,
            timeout=BENCHMARK_SYNTH_TIMEOUT_S,
        )
        synth_time = time.monotonic() - start_time
        gate_count = _parse_gate_count(result.stdout)

        if result.returncode != 0:
            cache.close()
            return {
                "component": component,
                "params": params,
                "gate_count": 0,
                "area_um2": 0,
                "max_clock_mhz": 0,
                "worst_slack_ns": None,
                "synthesis_time_s": round(synth_time, 2),
                "cached": False,
                "error": f"Yosys failed: {result.stderr[-500:]}",
            }

    except FileNotFoundError:
        cache.close()
        return {
            "component": component,
            "params": params,
            "gate_count": 0,
            "area_um2": 0,
            "max_clock_mhz": 0,
            "worst_slack_ns": None,
            "synthesis_time_s": 0,
            "cached": False,
            "error": "Yosys not installed",
        }
    except subprocess.TimeoutExpired:
        cache.close()
        return {
            "component": component,
            "params": params,
            "gate_count": 0,
            "area_um2": 0,
            "max_clock_mhz": 0,
            "worst_slack_ns": None,
            "synthesis_time_s": BENCHMARK_SYNTH_TIMEOUT_S,
            "cached": False,
            "error": f"Synthesis timed out after {BENCHMARK_SYNTH_TIMEOUT_S}s",
        }

    # Optional STA -- report honestly when tools are missing
    worst_slack_ns = None
    max_clock_mhz = None  # None means "not measured", distinct from 0.0
    sta_skipped_reason = None

    if skip_sta:
        sta_skipped_reason = "skipped by caller"
    elif not netlist_path.exists():
        sta_skipped_reason = "no netlist produced"
    elif not Path(liberty).exists():
        sta_skipped_reason = "liberty file not found"
    else:
        sdc_content = (
            f"create_clock -name clk -period {clock_period_ns} [get_ports clk]\n"
            f"set_input_delay -clock clk {clock_period_ns * 0.2} [all_inputs]\n"
            f"set_output_delay -clock clk {clock_period_ns * 0.2} [all_outputs]\n"
        )
        sdc_path = out_dir / f"{module_name}.sdc"
        sdc_path.write_text(sdc_content)

        sta_script = (
            f"read_liberty {liberty}\n"
            f"read_verilog {netlist_path}\n"
            f"link_design {module_name}\n"
            f"read_sdc {sdc_path}\n"
            f"report_checks -path_delay max -digits 4\n"
            f"report_wns\n"
            f"exit\n"
        )

        try:
            sta_result = subprocess.run(
                ["sta", "-exit", "-"],
                input=sta_script,
                capture_output=True,
                text=True,
                timeout=BENCHMARK_STA_TIMEOUT_S,
            )
            sta_report = sta_result.stdout + "\n" + sta_result.stderr
            worst_slack_ns = _parse_sta_slack(sta_report)

            if worst_slack_ns is not None:
                max_period = clock_period_ns - worst_slack_ns
                if max_period > 0:
                    max_clock_mhz = round(1000.0 / max_period, 1)
                else:
                    max_clock_mhz = 0.0
            else:
                sta_skipped_reason = "could not parse slack from STA output"
        except FileNotFoundError:
            sta_skipped_reason = "OpenSTA (sta) not installed"
        except subprocess.TimeoutExpired:
            sta_skipped_reason = f"STA timed out after {BENCHMARK_STA_TIMEOUT_S}s"

    benchmark_result = {
        "component": component,
        "params": params,
        "gate_count": gate_count,
        "area_um2": 0,  # Would need detailed area report
        "max_clock_mhz": max_clock_mhz,
        "worst_slack_ns": worst_slack_ns,
        "synthesis_time_s": round(synth_time, 2),
        "cached": False,
    }
    if sta_skipped_reason:
        benchmark_result["sta_skipped"] = sta_skipped_reason

    # Cache the result
    cache.store(component, params, pdk_config.name, target_clock_mhz, benchmark_result)
    cache.close()

    return benchmark_result


async def characterize_pdk(
    pdk_config: PDKConfig,
    target_clock_mhz: float = 50.0,
    project_root: str = ".",
) -> dict[str, Any]:
    """Run standard benchmark suite against a PDK.

    Results are cached so subsequent calls return immediately.

    Args:
        pdk_config: PDK configuration.
        target_clock_mhz: Target clock frequency.
        project_root: Project root for output paths.

    Returns:
        Dict mapping benchmark names to results.
    """
    benchmarks = [
        ("multiplier", {"width": 8}),
        ("multiplier", {"width": 16}),
        ("multiplier", {"width": 32}),
        ("fifo", {"depth": 64, "width": 8}),
        ("fifo", {"depth": 1024, "width": 32}),
        ("sram_array", {"depth": 256, "width": 8}),
        ("sram_array", {"depth": 4096, "width": 8}),
        ("fft_butterfly", {"radix": 2, "width": 16}),
        ("counter", {"width": 16}),
        ("counter", {"width": 32}),
    ]

    results = {}
    for component, params in benchmarks:
        key = f"{component}_{params}"
        results[key] = await run_benchmark(
            component, params, pdk_config, target_clock_mhz, project_root
        )

    return results
