"""The opt-in supervisor (CORESMITH_SUPERVISOR=1) hands the frontend off once
the stage machine reaches ``blocks`` and never launches, resumes or decides
for an Architect; native-session failures are classified from the runner's
diagnostics, never from the model's text."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestrator.architect.outcomes import outcome
from orchestrator.daemon import supervisor as sup
from orchestrator.state_store.project_db import open_project


@pytest.mark.parametrize('message', ['Billing verification failed', 'invalid API key', 'rate limit'])
def test_provider_failure_is_not_model_no_progress(message):
    assert outcome({'ok': False, 'rc': 1, 'error': message})['kind'] == 'provider_blocked'


def test_process_failure_is_distinct():
    assert outcome({'ok': False, 'rc': -9, 'stderr': 'timeout'})['kind'] == 'tool_failed'
    assert outcome({'ok': True, 'rc': 0})['kind'] == 'success'


def test_failure_reason_is_the_runner_diagnostic_not_the_model_text():
    """``sit()`` moves stderr to a file and keeps its tail on the result; the
    classifier reads that tail. The model's last message is never a reason."""
    res = {'ok': False, 'rc': -9, 'text': 'Let me diagnose by running refine with visible output:',
           'stderr_tail': '[invocation timed out after 14399s]'}
    out = outcome(res)
    assert out['kind'] == 'tool_failed' and out['reason'] == '[invocation timed out after 14399s]'
    assert 'diagnose' not in out['reason']
    out = outcome({'ok': False, 'rc': 124, 'timed_out': True, 'text': 'model text', 'stderr_tail': ''})
    assert out['status'] == 'timed_out' and 'model text' not in out['reason']
    out = outcome({'ok': False, 'rc': 2, 'text': 'model text'})
    assert out['reason'] == 'agent exited rc=2'


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('CORESMITH_SUPERVISOR', '1')
    db = open_project(tmp_path)
    monkeypatch.setattr('orchestrator.state_store.stages.status', lambda *_: {
        'stage': 'blocks', 'index': 1, 'stages': ['requirements', 'blocks']})
    return SimpleNamespace(_PROJECT_ROOT=str(tmp_path), _project_db=lambda: db,
        run_state=AsyncMock(return_value={'status': 'idle', 'values_empty': True}),
        run_start=AsyncMock(return_value={'started': True}), StartRequest=lambda: None,
        _daemon_log=lambda *a: None)


@pytest.mark.asyncio
async def test_handoff_once_across_checks(server):
    assert (await sup.tick(server))['actions'] == ['pipeline_started']
    await sup.tick(server)
    server.run_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_launch_is_durable_not_retried(server):
    server.run_start.side_effect = RuntimeError('tool missing')
    first = await sup.tick(server)
    second = await sup.tick(server)
    assert first['state'] == second['state'] == 'blocked'
    assert 'tool missing' in second['reason']
    server.run_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_design_decision_or_launch_on_park(server):
    server.run_state.return_value = {'status': 'interrupted', 'interrupts': [{'id': 'i'}]}
    assert (await sup.tick(server))['state'] == 'awaiting_decision'
    server.run_start.assert_not_called()


@pytest.mark.asyncio
async def test_provider_block_prevents_handoff(server):
    server._project_db().set_flag('agent_failure', {'kind': 'provider_blocked', 'reason': 'billing'})
    assert (await sup.tick(server))['reason'] == 'billing'
    server.run_start.assert_not_called()


@pytest.mark.asyncio
async def test_frontend_completion_is_not_full_signoff(server):
    server.run_state.return_value = {'pipeline_done': True, 'frontend_outcome': 'complete', 'status': 'done'}
    result = await sup.tick(server)
    assert result['state'] == 'frontend_complete'
    assert 'frontend' in result['reason']
    server.run_start.assert_not_called()


def test_user_pause_is_not_restarted():
    assert sup.classify({'status': 'paused'}, None)[0] == 'blocked'


@pytest.mark.asyncio
async def test_supervisor_never_launches_an_architect_before_blocks(server, monkeypatch):
    """Before ``blocks`` the Architect (outside the engine) drives ``stage
    next``; the supervisor spawns nothing and launches no pipeline."""
    monkeypatch.setattr('orchestrator.state_store.stages.status', lambda *_: {
        'stage': 'requirements', 'index': 0, 'stages': ['requirements', 'blocks']})
    import subprocess
    spawned = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(RuntimeError('spawn')))
    for _ in range(3):
        result = await sup.tick(server)
        assert result['state'] == 'running' and result['actions'] == []
        assert 'waiting for the Architect' in result['reason']
    assert spawned == []
    server.run_start.assert_not_called()
    assert server._project_db().get_flag('supervisor_architect') is None
    assert not hasattr(sup, '_architect')


@pytest.mark.asyncio
async def test_stale_architect_status_does_not_influence_the_run(server, tmp_path):
    """A status.json left by an older engine's Architect loop is ignored."""
    adir = tmp_path / '.coresmith' / 'architect'
    adir.mkdir(parents=True)
    (adir / 'status.json').write_text('{"state": "no_progress", "stop_reason": "2 consecutive sittings"}')
    result = await sup.tick(server)
    assert result['state'] == 'running' and result['actions'] == ['pipeline_started']


@pytest.mark.asyncio
async def test_cluster_worker_failure_is_infrastructure_state(server, tmp_path):
    cdir = tmp_path / '.coresmith' / 'clusters' / 'cpu'
    cdir.mkdir(parents=True)
    (cdir / 'status.json').write_text('{"state": "tool_failed", "stop_reason": "[invocation timed out after 60s]"}')
    result = await sup.tick(server)
    assert result['state'] == 'blocked' and result['failure']['worker'] == 'cpu'
    assert 'timed out' in result['reason']
    server.run_start.assert_not_called()


@pytest.mark.parametrize("pid", [0, -1, None, "invalid"])
def test_missing_daemon_pid_is_never_alive(pid):
    import runpy
    from pathlib import Path
    cli = runpy.run_path(str(Path(__file__).resolve().parents[2] / "bin/coresmith"))
    assert cli["_alive"](pid) is False
