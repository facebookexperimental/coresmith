# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unseeded graph build with real simulation, coverage, area and power.

The only implementation substitute is a deterministic RTL writer. The
Architect supplies a fixed acceptance testbench; no testbench worker is
allowed. Requires the same tools/environment as test_build_toolchain_smoke.
"""
import asyncio
import types
from pathlib import Path

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import (
    RTL,
    add_module_targets,
    make_build_lifecycle,
    ready_project,
)
from orchestrator.tests.test_build_toolchain_smoke import (
    FRD,
    HARNESS,
    MODEL,
    SINK_MODEL,
    _tools,
)

pytestmark = pytest.mark.slow

ACCEPTANCE = '''import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from orchestrator.harness.measure import record

async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.m_out.value) == 0
    dut.rst_n.value = 1

@cocotb.test()
async def test_count(dut):
    await reset(dut)
    for expected in range(1, 261):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.m_out.value) == expected % 256

@cocotb.test()
async def test_throughput(dut):
    await reset(dut)
    cycles = 0
    start = int(dut.m_out.value)
    for _ in range(16):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        cycles += 1
    operations = (int(dut.m_out.value) - start) % 256
    assert operations > 0
    record("PERF-TINY-1", cycles / operations, unit="cycles/op", test="test_throughput")
'''


@pytest.mark.parametrize('split_acceptance', [False, True])
def test_unseeded_targets_with_real_tools(tmp_path, monkeypatch, split_acceptance):
    lib, sta_dir, missing, sc_ok, vv = _tools()
    if missing or not lib or not sta_dir or not sc_ok or not vv or vv < (5, 36):
        pytest.skip(f"real toolchain incomplete: {missing}, liberty={lib}, sta={sta_dir}, SystemC={sc_ok}")
    import os

    from orchestrator.daemon import server
    from orchestrator.harness.tools import integrate as it
    from orchestrator.langgraph import pipeline_graph as pg
    from orchestrator.langgraph import pipeline_helpers as ph
    from orchestrator.state_store import stages

    monkeypatch.setenv('PATH', f"{sta_dir}:{os.environ['PATH']}")
    pdk = tmp_path / 'pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib'
    pdk.mkdir(parents=True)
    (pdk / Path(lib).name).symlink_to(lib)
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'pdk'))
    monkeypatch.setattr(ph, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(ph, 'LIBERTY_FILE', pdk / Path(lib).name)
    monkeypatch.setattr(ph, 'PDK_ROOT', tmp_path / 'pdk')
    monkeypatch.setattr(ph, '_LOG_DIR', tmp_path / '.coresmith/step_logs')
    db = ready_project(tmp_path, monkeypatch, with_rtl=False, with_model=False)
    monkeypatch.setenv('CORESMITH_PPA_GATE', '1')
    (tmp_path / 'arch/frd_spec.md').write_text(FRD)

    first = it.model_build(db, tmp_path)
    assert first['missing_models'] == ['sink', 'tiny'], first
    (tmp_path / 'model/tiny_model.cpp').write_text(MODEL)
    (tmp_path / 'model/sink_model.cpp').write_text(SINK_MODEL)
    for name, declaration in [('tiny', '  unsigned count;\n'), ('sink', '  unsigned beats;\n')]:
        header = tmp_path / 'model' / f'{name}_model.h'
        header.write_text(header.read_text().replace('  void run();', declaration + '  void run();', 1))
    model = it.model_build(db, tmp_path)
    assert model['build_ok'] and model['smoke_ok'], model
    (tmp_path / 'model/frd_eval/frd_eval.cpp').write_text(HARNESS)
    evaluated = it.model_eval(db, tmp_path)
    assert evaluated['ok'], evaluated

    ids = add_module_targets(db, tmp_path, power=True)
    db.edit_item(ids['area'], bound_max=1000.0)
    acceptance = tmp_path / ids['tb']
    acceptance.write_text(ACCEPTANCE)
    oracle_paths = [acceptance]
    if split_acceptance:
        protocol = acceptance.with_name('test_protocol.py')
        count_start = ACCEPTANCE.index('@cocotb.test()\nasync def test_count')
        rate_start = ACCEPTANCE.index('@cocotb.test()\nasync def test_throughput')
        protocol.write_text(ACCEPTANCE[:rate_start])
        acceptance.write_text(ACCEPTANCE[:count_start] + ACCEPTANCE[rate_start:])
        for verifier in db.verifiers(item_id='FUNC-001'):
            if verifier['kind'] == 'cocotb':
                db.remove_verifier(verifier['id'])
        db.add_verifier('FUNC-001', 'cocotb', path=str(protocol.relative_to(tmp_path)),
                        entry='test_count', block='tiny')
        oracle_paths.append(protocol)
    original_oracles = {p:B.file_sha256(p) for p in oracle_paths}
    assert not (tmp_path / 'rtl/tiny.v').exists()
    assert stages.module_ready(db, tmp_path, 'tiny') == []
    written = []

    async def implement(block, attempt, callbacks=None, **kwargs):
        target = tmp_path / 'rtl/tiny.v'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(RTL['tiny'])
        written.append(str(target))
        return {'rtl_path':str(target), 'verilog':target.read_text()}

    async def forbidden_tb_worker(*args, **kwargs):
        raise AssertionError('fixed acceptance must run without rewriting its oracle')

    monkeypatch.setattr(pg, 'generate_rtl', implement)
    monkeypatch.setattr(pg, 'generate_testbench', forbidden_tb_worker)
    monkeypatch.setattr(pg, 'create_golden_model_wrapper', lambda *args, **kwargs:None)
    lifecycle = make_build_lifecycle(tmp_path)
    monkeypatch.setattr(server, '_PROJECT_ROOT', str(tmp_path))
    monkeypatch.setattr(server, '_build', lifecycle)
    monkeypatch.setattr(server, '_pipeline', types.SimpleNamespace(task=None, thread_id='pipeline', status='idle'))
    monkeypatch.setattr(server, '_apply_run_env', lambda where:[])

    async def run():
        started = await server._start_module_build(server.BuildModuleRequest(module='tiny'), entry='build_module')
        assert isinstance(started, dict), getattr(started, 'body', started)
        bid = started['build_id']
        parks = []
        for _ in range(2400):
            row = B.get_build(db, bid)
            if row['status'] in ('parked',) + B.BUILD_TERMINAL:
                if row['status'] == 'parked':
                    parks, _ = await server.MB.live_parks(lifecycle, f'build-{bid}')
                break
            await asyncio.sleep(0.5)
        await lifecycle.cleanup()
        return bid, B.get_build(db, bid), parks

    bid, row, parks = asyncio.run(run())
    assert row['status'] == 'completed', f"{row.get('error')}\nparks={parks}\nresult={row.get('result')}"
    assert written and row['seed'] is None
    assert {p:B.file_sha256(p) for p in oracle_paths} == original_oracles
    candidate = B.latest_candidate(db, bid)
    assert candidate['outcome'] == 'feasible'
    targets = {t['id']:t for t in candidate['evaluation']['targets']}
    for name in ('perf', 'area', 'power'):
        target = targets[ids[name]]
        assert target['status'] == 'pass' and target['value'] > 0 and target['receipts'], target
        assert db.latest_check(ids[name], 'block_dv')['status'] == 'pass'
    canonical_netlist = tmp_path / 'syn/output/tiny/tiny_netlist.v'
    canonical_hash = B.file_sha256(canonical_netlist)
    assert canonical_hash
    for name in ('area', 'power'):
        assert all(measurement['receipt']['netlist_sha256'] == canonical_hash
                   for measurement in targets[ids[name]]['receipts']), targets[ids[name]]
    rows = Scoreboard(tmp_path).rows_for_build(bid)
    assert rows['coverage'][-1]['pct'] is not None
    assert rows['ppa'][-1]['power_basis'] == 'estimated' and rows['ppa'][-1]['power_mw'] > 0
    assert B.lineage(db, tmp_path, 'tiny')['modules']['tiny']['intended_workflow']
    print('\nUNSEEDED TARGET SMOKE OK:', {'build':bid, 'rtl_worker_calls':len(written),
          'targets':{iid:t['value'] for iid,t in targets.items()},
          'coverage_pct':rows['coverage'][-1]['pct'], 'oracle_unchanged':True,
          'acceptance_files':len(oracle_paths)})
