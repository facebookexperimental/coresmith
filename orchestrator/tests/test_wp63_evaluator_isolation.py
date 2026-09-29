"""Adapters cannot rewrite their inputs, trust baseline, or previous evaluations."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.harness import task_adapter as ta
from orchestrator.state_store import trust
from orchestrator.tests.candidate_fixtures import adopt


def project(tmp_path, monkeypatch, body=None):
    monkeypatch.setenv('CORESMITH_TRUST_DIR', str(tmp_path / 'owner-trust'))
    root = tmp_path / 'project'
    (root / 'inputs').mkdir(parents=True)
    top = root / 'top.v'
    top.write_text('module chip_top(); endmodule')
    (root / 'inputs/task.yaml').write_text('top: chip_top\n')
    adapter = root / 'inputs/task_adapter.py'
    adapter.write_text(body or 'CASES=["one"]\ndef grade(c,w):\n return {"cases":{"one":{"ok":True}}}\n')
    adopt(root, top)
    assert trust.write_oracle_manifest(root) is not None
    return root, top


def test_missing_bwrap_fails_closed_and_optout_is_explicit(tmp_path, monkeypatch):
    root, top = project(tmp_path, monkeypatch)
    monkeypatch.delenv('CORESMITH_ADAPTER_SANDBOX', raising=False)
    monkeypatch.delenv('CORESMITH_BWRAP', raising=False)
    monkeypatch.setattr(ta, '_SYSTEM_BWRAP', ())   # WP-77: no system bwrap either
    monkeypatch.setattr(shutil, 'which', lambda _: None)
    result = ta.run_task_adapter(str(root), str(top))
    assert result['kind'] == 'infrastructure_error'
    assert not result['passed']
    monkeypatch.setenv('CORESMITH_ADAPTER_SANDBOX', 'none')
    assert ta.run_task_adapter(str(root), str(top))['passed']


def test_sandbox_argv_and_private_work_policy(tmp_path, monkeypatch):
    root, top = project(tmp_path, monkeypatch)
    monkeypatch.delenv('CORESMITH_ADAPTER_SANDBOX', raising=False)
    monkeypatch.delenv('CORESMITH_BWRAP', raising=False)
    monkeypatch.setattr(ta, '_SYSTEM_BWRAP', ())   # WP-77: PATH is the fallback here
    monkeypatch.setattr(shutil, 'which', lambda _: '/fake/bwrap')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        # The fake runner writes the same minimal declared receipt as the real one.
        i = argv.index(str(ta._RUNNER))
        Path(argv[i+3]).write_text(json.dumps({'declared_cases':['one'], 'cases':{'one':{'ok':True}}}))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', run)
    assert ta.run_task_adapter(str(root), str(top))['passed']
    argv, kw = calls[0]
    assert argv[:4] == ['/fake/bwrap', '--ro-bind', '/', '/']
    for flag in ['--unshare-net', '--die-with-parent']:
        assert flag in argv
    for flag, path in [('--tmpfs','/tmp'),('--dev','/dev'),('--proc','/proc')]:
        assert argv[argv.index(flag)+1] == path
    i = argv.index('--bind')
    assert argv[i+1] == argv[i+2]
    assert Path(argv[i+1]).name == 'work'
    assert kw['env']['TMPDIR'] == '/tmp'


def test_adapter_cannot_write_outside_workdir(tmp_path, monkeypatch):
    """Two boundaries: a write outside the project never reaches the host, and
    the project itself is read-only so an adapter cannot edit the chip, its
    inputs or its own manifest while grading them (WP-63, tightened by WP-79).
    """
    bwrap = shutil.which('bwrap')
    if not bwrap:
        pytest.skip('bubblewrap unavailable')
    probe = subprocess.run([bwrap,'--ro-bind','/','/','--dev','/dev','--proc','/proc','--tmpfs','/tmp','--unshare-net','--die-with-parent',sys.executable,'-c','pass'], capture_output=True)
    if probe.returncode:
        pytest.skip('bubblewrap namespace unavailable: '+probe.stderr.decode()[:200])

    # 1. a write aimed outside the project is contained
    marker = tmp_path / 'outside.txt'
    root, top = project(tmp_path, monkeypatch, 'from pathlib import Path\nCASES=["one"]\ndef grade(c,w):\n Path('+repr(str(marker))+').write_text("tamper")\n return {"cases":{"one":{"ok":True}}}\n')
    monkeypatch.delenv('CORESMITH_ADAPTER_SANDBOX', raising=False)
    result = ta.run_task_adapter(str(root), str(top))
    assert not marker.exists(), 'adapter write escaped to the host'
    assert result['completed_cases'] == 1

    # 2. a write into the project itself fails, and the engine reports it
    #    rather than recording a pass
    shutil.rmtree(tmp_path / 'project')
    (tmp_path / 'owner-trust').exists() and shutil.rmtree(tmp_path / 'owner-trust')
    inside = tmp_path / 'project' / 'inputs' / 'task.yaml'
    root, top = project(tmp_path, monkeypatch, 'from pathlib import Path\nCASES=["one"]\ndef grade(c,w):\n Path('+repr(str(inside))+').write_text("top: other\\n")\n return {"cases":{"one":{"ok":True}}}\n')
    result = ta.run_task_adapter(str(root), str(top))
    assert not result['passed'], 'a tampering adapter must not report a pass'
    assert inside.read_text() == 'top: chip_top\n', 'project was modified'


def test_baseline_is_external_and_deletion_is_an_error(tmp_path, monkeypatch):
    root, top = project(tmp_path, monkeypatch)
    manifest = trust._manifest_path(root)
    assert not manifest.is_relative_to(root)
    manifest.unlink()
    result = trust.check_oracle_manifest(root)
    assert not result['ok']
    assert result['violation']
    assert not ta.run_task_adapter(str(root), str(top))['passed']


def test_receipt_and_log_history_does_not_overwrite(tmp_path, monkeypatch):
    root, top = project(tmp_path, monkeypatch)
    monkeypatch.setenv('CORESMITH_ADAPTER_SANDBOX', 'none')
    first = ta.run_task_adapter(str(root), str(top))
    keep = Path(first['captured_dir'])
    before = {p: p.read_bytes() for p in [keep/'receipt.json', keep/'adapter.log']}
    second = ta.run_task_adapter(str(root), str(top))
    assert first['passed'] and second['passed']
    assert first['candidate_sha'] == second['candidate_sha']
    assert first['captured_dir'] != second['captured_dir']
    assert all(p.read_bytes() == data for p, data in before.items())


def test_sandbox_rebinds_an_engine_under_tmp(monkeypatch, tmp_path):
    """An engine checkout under /tmp (worktree, CI scratch) is masked by the
    sandbox's tmpfs; it must be re-bound read-only or the runner is missing."""
    monkeypatch.setenv('CORESMITH_ADAPTER_SANDBOX', 'bwrap')
    monkeypatch.setattr(ta, '_find_bwrap', lambda: '/fake/bwrap')
    engine = str(Path(ta.__file__).resolve().parents[2])
    argv = ta._sandbox_argv(['x'], tmp_path / 'work', str(tmp_path / 'proj'))
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == '--ro-bind']
    assert (engine in binds) == engine.startswith('/tmp/')
    monkeypatch.setattr(ta.Path, 'resolve', lambda self: Path('/tmp/eng/orchestrator/harness/task_adapter.py') if str(self).endswith('task_adapter.py') else Path(str(self)))
    argv = ta._sandbox_argv(['x'], tmp_path / 'work', str(tmp_path / 'proj'))
    assert argv[argv.index('--tmpfs') + 2:argv.index('--tmpfs') + 3] != ['--bind'] and '/tmp/eng' in argv
