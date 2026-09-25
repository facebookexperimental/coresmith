# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A scratch simulator can setsid and outlive its CLI; cleanup stays scoped."""
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from orchestrator.langchain.agents import coresmith_llm as llm

pytestmark = pytest.mark.skipif(
    not Path("/proc").is_dir() or not hasattr(os, "pidfd_open")
    or not hasattr(signal, "pidfd_send_signal"), reason="Linux pidfd ownership test")


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


@pytest.mark.parametrize("pause", [False, True, 'legacy_cancel'])
@pytest.mark.parametrize("stopped", [False, True])
def test_detached_child_reaped_after_parent_exit_without_touching_other_call(tmp_path, pause, stopped):
    scope = uuid.uuid4().hex
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             env={**os.environ, llm._PROCESS_SCOPE_ENV: uuid.uuid4().hex},
                             start_new_session=True)
    child_pid = None
    parent = None
    try:
        # The grandchild creates a separate process group and outlives its
        # direct parent, reproducing the failed PGID-only cleanup.
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
                "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "print(p.pid,flush=True)")
        parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                  text=True, start_new_session=True,
                                  env={**os.environ, llm._PROCESS_SCOPE_ENV: scope})
        parent._coresmith_process_scope = scope
        child_pid = int(parent.stdout.readline())
        parent.wait(timeout=5)
        assert alive(child_pid) and os.getpgid(child_pid) != parent.pid
        if stopped:
            os.kill(child_pid, signal.SIGSTOP)
        if pause == 'legacy_cancel':
            llm._register_process(parent)
            assert llm.kill_active_cli_processes() == 1
        elif pause:
            llm._register_process(parent)
            assert llm.reap_active_cli_processes(grace_s=.1) == 1
        else:
            llm._reap_process_group(parent, parent.pid, grace_s=.1)
        deadline = time.monotonic() + 3
        while alive(child_pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not alive(child_pid)
        assert other.poll() is None
        assert llm._PROCESS_SCOPE_ENV not in os.environ
    finally:
        llm._unregister_process()
        for pid in [child_pid, parent.pid if parent else None, other.pid]:
            if pid and alive(pid):
                os.kill(pid, signal.SIGKILL)
        other.wait(timeout=5)
