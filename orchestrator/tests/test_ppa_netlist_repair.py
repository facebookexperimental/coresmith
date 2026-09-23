# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Mapped FF buffering, honest failure, and same-netlist PPA accounting."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.langgraph import ppa_check as pc
from orchestrator.pdk import registry
from orchestrator.pdk.base import CheckResult, ToolRequest, ToolResult
from orchestrator.pdk.deployments import sky130


@pytest.fixture
def repair_request(tmp_path, monkeypatch):
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'pdk'))
    dep = sky130.Sky130Deployment()
    for p in (dep.tech_lef, dep.cell_lef, dep.liberty):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('fixture')
    netlist = tmp_path / 'input.v'
    netlist.write_text('module core(input clk); endmodule')
    req = ToolRequest('repair_netlist', 'core',
                      inputs={'netlist': netlist, 'liberty': dep.liberty},
                      out_dir=tmp_path / 'repair', params={'clock_ns': 15.625},
                      timeout_s=17)
    return dep, req


@pytest.mark.parametrize('failure', ['missing_binary', 'timeout', 'rc', 'no_output', 'no_marker'])
def test_repair_cannot_accept_stale_artifact(repair_request, monkeypatch, failure):
    dep, req = repair_request
    req.out_dir.mkdir()
    output = req.out_dir / 'repaired.v'
    output.write_text('stale')

    def run(cmd, timeout, cwd):
        assert timeout == 17 and not output.exists()
        if failure == 'missing_binary':
            return 127, '', '', 'tool not found'
        if failure == 'timeout':
            return 124, '', '', 'timeout'
        if failure != 'no_output':
            output.write_text('fresh')
        return (1 if failure == 'rc' else 0,
                '' if failure == 'no_marker' else 'CORESMITH_REPAIR_DONE', '', '')

    monkeypatch.setattr(sky130, '_run_cmd', run)
    result = dep.tool('repair_netlist').run(req)
    assert not result.ok and 'netlist' not in result.artifacts
    assert result.log_path.exists()


def test_missing_pdk_never_launches_repair(repair_request, monkeypatch):
    dep, req = repair_request
    dep.tech_lef.unlink()
    monkeypatch.setattr(sky130, '_run_cmd', lambda *a, **k: pytest.fail('must not launch'))
    result = dep.tool('repair_netlist').run(req)
    assert not result.ok and not result.tool_ok
    assert str(dep.tech_lef) in result.checks[0].details


def test_repair_uses_requested_corner_and_deployment_settings(repair_request, monkeypatch):
    dep, req = repair_request
    corner = req.inputs['liberty'].with_name('slow corner.lib')
    corner.write_text('fixture')
    req.inputs['liberty'] = corner
    dep.pdk.site_name = 'custom_site'
    dep.pdk.pnr.max_fanout = 7
    monkeypatch.setattr(dep, 'resolve_openroad_bin', lambda: '/configured/openroad')

    def run(cmd, **kwargs):
        assert cmd[0] == '/configured/openroad'
        script = Path(cmd[-1]).read_text()
        assert f'read_liberty "{corner}"' in script
        assert '-site "custom_site"' in script
        assert 'set_max_fanout 7' in script
        (req.out_dir / 'repaired.v').write_text('fresh')
        return 0, 'CORESMITH_REPAIR_DONE', '', ''

    monkeypatch.setattr(sky130, '_run_cmd', run)
    result = dep.tool('repair_netlist').run(req)
    assert result.ok and result.artifacts['netlist'].read_text() == 'fresh'


def fake_measurement_tools(tmp_path, monkeypatch, repair_status='repaired', buf_wns=5):
    src, lib = tmp_path / 'core.v', tmp_path / 'cells.lib'
    src.write_text('module core(input clk); endmodule')
    lib.write_text('library(test){}')
    seen = []

    def run(cmd, **kwargs):
        script = Path(cmd[-1])
        tag = script.parent.name
        if script.name == 'syn.ys':
            script.with_name('netlist.v').write_text(f'module core; // {tag}\nendmodule')
            stdout = ''
        elif script.name == 'stat.ys':
            # The repaired artifact, not the original map, is priced.
            is_repaired = 'repaired.v' in script.read_text()
            area = 200 if is_repaired else 80
            stdout = (f'=== core ===\n5 sky130_fd_sc_hd__dfxtp_1\n'
                      f'Number of cells: 8\nChip area for module core: {area}.0\n')
        else:
            assert script.name == 'sta.tcl'
            seen.append(script.read_text())
            stdout = f'CORESMITH_WNS {buf_wns if tag == "buf" else -10}\n'
        return subprocess.CompletedProcess(cmd, 0, stdout, '')

    def repair(req):
        req.out_dir.mkdir()
        output = req.out_dir / 'repaired.v'
        output.write_text('module core; // repaired\nendmodule')
        if repair_status == 'failed':
            return ToolResult.from_checks(tool_ok=False,
                checks=[CheckResult('repair', 'not_run', details='timeout')])
        return ToolResult.from_checks(tool_ok=True, checks=[], artifacts={'netlist': output})

    tool = None if repair_status == 'unavailable' else SimpleNamespace(run=repair)
    monkeypatch.setattr(registry, 'get_deployment', lambda: SimpleNamespace(tool=lambda name: tool))
    monkeypatch.setattr(pc.shutil, 'which', lambda name: name)
    monkeypatch.setattr(pc.subprocess, 'run', run)
    result = pc.run_maxfanout_buffered_sta(str(src), str(lib), 'core', 15.625,
                                          report_dir=tmp_path / 'reports')
    return result, seen


def test_selected_timing_prices_and_preserves_repaired_netlist(tmp_path, monkeypatch):
    result, seen = fake_measurement_tools(tmp_path, monkeypatch)
    assert result['sta_ok'] and result['wns_ns'] == 5
    assert result['selected_variant'] == 'buf' and result['repair_status'] == 'repaired'
    assert result['chip_area_um2'] == 200 and result['ff_count'] == 5
    assert '// repaired' in Path(result['netlist_path']).read_text()
    assert 'repaired.v' in seen[1]


@pytest.mark.parametrize('status', ['failed', 'unavailable'])
def test_repair_failure_keeps_honest_measured_fallback(tmp_path, monkeypatch, status):
    result, seen = fake_measurement_tools(tmp_path, monkeypatch, status, buf_wns=-12)
    assert result['sta_ok'] and result['wns_ns'] == -10
    assert result['selected_variant'] == 'base' and result['chip_area_um2'] == 80
    assert result['repair_status'] == status and status in result['detail']
    assert 'repaired.v' not in seen[1]


@pytest.mark.parametrize('base_wns, expected_area, expected_ff', [(-10, 200, 9), (10, 80, 4)])
def test_pipeline_gates_on_one_candidates_timing_and_area(tmp_path, monkeypatch,
                                                        base_wns, expected_area, expected_ff):
    from orchestrator.langgraph import pipeline_graph as pg
    rtl, lib, sdc = tmp_path / 'core.v', tmp_path / 'cells.lib', tmp_path / 'core.sdc'
    rtl.write_text('module core(input clk); endmodule')
    lib.write_text('fixture')
    sdc.write_text('create_clock -period 15.625 [get_ports clk]')
    spec = tmp_path / 'arch/uarch_specs/core.md'
    spec.parent.mkdir(parents=True)
    spec.write_text('area_budget_um2: 100\nflip_flop_budget: 100\n')
    monkeypatch.setattr(pc, 'probe_synth_generic', lambda *a, **k: {'elaborated': True, 'logic_ff': 4})
    monkeypatch.setattr(pc, 'synth_cell_gate_enabled', lambda: False)
    monkeypatch.setattr(pc, 'logic_depth_gate_enabled', lambda: False)
    monkeypatch.setattr(pc, 'sta_maxfanout_enabled', lambda: True)
    monkeypatch.setattr(pc, 'run_pre_layout_sta', lambda *a, **k: {'wns_ns': base_wns})
    monkeypatch.setattr(pc, 'run_maxfanout_buffered_sta', lambda *a, **k: {
        'sta_ok': True, 'wns_ns': 5, 'chip_area_um2': 200, 'ff_count': 9,
        'repair_status': 'repaired', 'selected_variant': 'buf',
        'netlist_path': 'selected.v', 'report_path': 'selected.rpt', 'netlist_sha256': 'abc'})
    ok, reasons, meta = pg._evaluate_ppa_gate(str(tmp_path), 'core', str(rtl), {
        'netlist_path': str(rtl), 'sdc_path': str(sdc), 'liberty_path': str(lib),
        'ff_count': 4, 'chip_area_um2': 80}, require_gate_flag=False)
    assert meta['area_um2'] == expected_area and meta['ff'] == expected_ff
    assert meta['wns_ns'] == max(base_wns, 5)
    assert ok is (expected_area == 80)
    if expected_area == 200:
        assert any('area' in r.lower() for r in reasons)
        assert meta['sta_report_path'] == 'selected.rpt'


def test_real_liberty_stat_cell_total_is_not_zero():
    from orchestrator.pdk.checkers import SynthStatChecker
    metrics = SynthStatChecker.parse_text('''=== core ===
    17153        - wires
    21633 2.76E+05 cells
      102 2.04E+03 sky130_fd_sc_hd__dfxtp_1
     6124 1.84E+05 sky130_fd_sc_hd__edfxtp_1
   Chip area for module core: 275958.416000
''')
    assert metrics == {'cells': 21633, 'ff_count': 6226, 'chip_area_um2': 275958.416}


def test_repair_cli_carries_liberty_and_clock(tmp_path, monkeypatch):
    from orchestrator.harness.cli_tool import _build_request
    monkeypatch.chdir(tmp_path)
    req = _build_request('repair_netlist', SimpleNamespace(
        design='core', netlist='mapped.v', liberty='slow corner.lib',
        clock_ns=15.625, clock_port='clock', out_dir='repair', timeout_s=20))
    assert req.input('liberty') == tmp_path / 'slow corner.lib'
    assert req.params == {'clock_ns': 15.625, 'clock_port': 'clock'}


def test_sta_failure_does_not_accept_printed_slack(tmp_path, monkeypatch):
    def run(cmd, **kwargs):
        if cmd[0] == 'yosys':
            Path(cmd[-1]).with_name('netlist.v').write_text('module core; endmodule')
            return subprocess.CompletedProcess(cmd, 0, '', '')
        return subprocess.CompletedProcess(cmd, 1, 'CORESMITH_WNS 10\n', 'error')
    monkeypatch.setattr(pc.subprocess, 'run', run)
    wns, detail = pc._measure_wns_from_rtl([], 'lib', tmp_path, 'base', False,
                                        15.625, 'core', 'clk', 'yosys', 'sta', 30)
    assert wns is None and 'rc=1' in detail


@pytest.mark.e2e
def test_real_mapped_ff_fanout_closes_without_rtl_changes(tmp_path, monkeypatch):
    """Opt-in Sky130 regression for the direct-D-pin load missed by ABC."""
    import collections
    import json
    import os
    import shutil

    pdk_root = os.environ.get('CORESMITH_TEST_PDK_ROOT')
    if not pdk_root:
        pytest.skip('set CORESMITH_TEST_PDK_ROOT and provide yosys/sta/OpenROAD')
    monkeypatch.setenv('PDK_ROOT', pdk_root)
    dep = sky130.Sky130Deployment()
    monkeypatch.setattr(registry, 'get_deployment', lambda: dep)
    assert dep.liberty.is_file() and dep.cell_lef.is_file() and dep.tech_lef.is_file()
    assert shutil.which('yosys') and shutil.which('sta')
    rtl = tmp_path / 'fanout.v'
    rtl.write_text('''module fanout(input clk, input d, input we,
        input [10:0] addr, output reg [1535:0] q);
reg data_q;
always @(posedge clk) data_q <= d;
genvar i;
generate for(i=0;i<1536;i=i+1) begin: bank
always @(posedge clk) if(we && addr==i) q[i] <= data_q;
end endgenerate
endmodule
''')
    result = pc.run_maxfanout_buffered_sta(str(rtl), str(dep.liberty), 'fanout', 15.625,
                                          report_dir=tmp_path / 'reports')
    assert result['base_wns_ns'] < 0
    assert result['repair_status'] == 'repaired' and result['wns_ns'] > 0
    assert result['ff_count'] == 1537 and result['cells'] > 1537
    assert result['chip_area_um2'] > 0
    netlist = Path(result['netlist_path'])
    parsed = tmp_path / 'mapped.json'
    script = tmp_path / 'mapped.ys'
    script.write_text(f'read_liberty -lib "{dep.liberty}"\n'
                      f'read_verilog "{netlist}"\nwrite_json "{parsed}"\n')
    subprocess.run(['yosys', '-Q', '-T', str(script)], capture_output=True, check=True, timeout=60)
    module = json.loads(parsed.read_text())['modules']['fanout']
    sinks = collections.Counter()
    for cell in module['cells'].values():
        for port, bits in cell['connections'].items():
            if cell['port_directions'][port] == 'input' and port != 'CLK':
                sinks.update(b for b in bits if isinstance(b, int))
    assert max(sinks.values()) <= dep.pdk.pnr.max_fanout


def test_repair_does_not_delete_its_input(repair_request, monkeypatch):
    dep, req = repair_request
    req.out_dir.mkdir()
    req.inputs['netlist'] = req.out_dir / 'repaired.v'
    req.inputs['netlist'].write_text('input must survive')
    monkeypatch.setattr(sky130, '_run_cmd', lambda *a, **k: pytest.fail('must not launch'))
    result = dep.tool('repair_netlist').run(req)
    assert not result.ok
    assert req.inputs['netlist'].read_text() == 'input must survive'
