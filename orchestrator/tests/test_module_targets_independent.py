"""Coordinator checks of evidence boundaries, independent of the implementation."""
import pytest

from orchestrator.state_store import module_targets as MT


def target(measure, kind='eda', unit='um2'):
    binding = {'kind':kind, 'entry':measure, 'block':'tiny', 'path':'tb/test_tiny.py', 'args':{}}
    return {'id':'PERF-TINY-1', 'required':True, 'metric':measure, 'unit':unit,
            'bound_min':None, 'bound_max':100, 'bindings':[binding]}


def eda_measurement(t, value):
    return {MT.measure_key(t['bindings'][0]):value}


@pytest.mark.parametrize('measure,unit', [('area_um2','um2'), ('power_mw','mW')])
@pytest.mark.parametrize('receipt', [None, {}, {'note':'not an execution receipt'}])
def test_positive_eda_number_without_tool_receipt_cannot_close(measure, unit, receipt):
    """A parsed/returned scalar alone does not establish execution or provenance."""
    t = target(measure, unit=unit)
    alloc = {'module':'tiny', 'targets':[t], 'functional':[]}
    result = MT.evaluate(alloc, sim=None, eda=eda_measurement(t, {'value':1.0, 'receipt':receipt}))
    assert result['outcome'] == 'unmeasured', result
    assert any('tool receipt' in reason for reason in result['targets'][0]['reasons']), result


@pytest.mark.parametrize('receipt', [None, {}, {'note':'not an execution receipt'}])
def test_measurement_and_test_name_without_simulation_receipt_cannot_close(receipt):
    """A JSONL value plus a claimed test pass is not a simulation receipt."""
    alloc = {'module':'tiny', 'targets':[target('test_rate', kind='cocotb', unit='cycles/op')],
             'functional':[], 'acceptance':{'module':'test_tiny'}}
    sim = {'receipt':receipt, 'tests':{'test_rate':'pass'},
           'rows':[{'item':'PERF-TINY-1','test':'test_rate','module':'test_tiny',
                    'value':1.0,'unit':'cycles/op'}]}
    result = MT.evaluate(alloc, sim=sim, eda=None)
    assert result['outcome'] == 'unmeasured', result


@pytest.mark.parametrize('receipt', [None, {}, {'note':'not an execution receipt'}])
def test_functional_pass_without_execution_receipt_cannot_close(receipt):
    alloc = {'module':'tiny', 'targets':[], 'functional':[
        {'id':'FUNC-TINY-1', 'required':True,
         'bindings':[{'kind':'cocotb','entry':'test_behavior','path':'tb/test_tiny.py','args':{}}]}]}
    result = MT.evaluate(alloc, sim={'receipt':receipt,'tests':{'test_behavior':'pass'}}, eda=None)
    assert result['outcome'] == 'unmeasured', result


def test_unavailable_advisory_measurement_does_not_refuse_a_build():
    """An optional target is reported as unavailable without becoming a gate."""
    optional = target('power_mw', unit='mW')
    optional['required'] = False
    alloc = {'module':'tiny', 'targets':[optional], 'functional':[]}
    assert MT.measurability(alloc, liberty_present=False, synth_generic=False,
                           sta_problem='OpenSTA unavailable') == []
    result = MT.evaluate(alloc, sim=None, eda=eda_measurement(optional, {'error':'OpenSTA unavailable'}))
    assert result['outcome'] == 'feasible'
    assert result['targets'][0]['status'] == 'unmeasured'


def test_separate_acceptance_files_are_inputs_of_the_same_module(tmp_path, monkeypatch):
    """Protocol and throughput checks can be bound in separate source files."""
    from orchestrator.state_store import builds as B
    from orchestrator.tests.build_fixtures import add_module_targets, ready_project

    db = ready_project(tmp_path, monkeypatch)
    add_module_targets(db, tmp_path)
    other = tmp_path / 'tb/cocotb/test_protocol.py'
    other.write_text('import cocotb\n@cocotb.test()\nasync def test_reset(dut):\n    assert True\n')
    for verifier in db.verifiers(item_id='FUNC-001'):
        if verifier['kind'] == 'cocotb':
            db.remove_verifier(verifier['id'])
    db.add_verifier('FUNC-001', 'cocotb', path=str(other.relative_to(tmp_path)),
                    entry='test_reset', block='tiny')
    allocation = MT.allocation(db, tmp_path, 'tiny')
    assert allocation['problems'] == [], allocation['problems']
    before = B.module_inputs(db, tmp_path, 'tiny')
    other.write_text(other.read_text() + '\n# oracle revision\n')
    after = B.module_inputs(db, tmp_path, 'tiny')
    assert any(reason.startswith('acceptance:') for reason in B.stale_reasons(before, after))


def test_an_unobservable_tool_is_not_a_positive_identity_match():
    from orchestrator.state_store import builds as B

    recorded = {'path':'/tools/sta', 'sha256':'a' * 64, 'version':'OpenSTA 3.1.0', 'error':None}
    unavailable = {'path':None, 'sha256':None, 'version':None, 'error':'sta not on PATH'}
    assert B._same_tool(recorded, unavailable) is not True


def test_area_report_without_its_netlist_and_library_cannot_close(tmp_path):
    """Required measurement inputs must exist, even if absent when captured."""
    from orchestrator.state_store import builds as B

    report = tmp_path / 'synthesis.rpt'
    report.write_text("Chip area for module 'tiny': 1.0\n")
    receipt = {'kind':MT.AREA_RECEIPT, 'tool':'yosys stat -liberty', 'area_scope':'total',
               'report_path':str(report), 'report_sha256':B.file_sha256(report),
               'netlist_path':str(tmp_path / 'missing.v'), 'netlist_sha256':None,
               'liberty':str(tmp_path / 'missing.lib'), 'liberty_sha256':None,
               'macro_libs':[], 'macros':[], 'macros_unresolved':[]}
    t = target('area_um2')
    alloc = {'module':'tiny', 'targets':[t], 'functional':[]}
    result = MT.evaluate(alloc, sim=None, eda=eda_measurement(t, {'value':1.0, 'receipt':receipt}))
    assert result['outcome'] == 'unmeasured', result
