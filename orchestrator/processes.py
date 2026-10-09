"""Owned subprocesses for model and EDA calls. Cancellation fences future launches."""
from __future__ import annotations

import contextvars
import fcntl
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_owner = contextvars.ContextVar("process_owner", default="")
_lock = threading.RLock()
_active: dict[int, tuple[subprocess.Popen, str]] = {}
_cancelled: set[str] = set()
_local = threading.local()


def begin_owner(label: str) -> str:
    owner = label + ":" + uuid.uuid4().hex
    _owner.set(owner)
    return owner


def fence(owner: str) -> None:
    with _lock:
        _cancelled.add(owner)


def register(proc):
    with _lock:
        owner = _owner.get()
        if owner in _cancelled:
            _reap_process_group(proc, proc.pid, grace_s=1)
            raise RuntimeError("job cancelled; refusing a new child")
        _active[proc.pid] = (proc, owner)
        _local.pid = proc.pid


def unregister(proc=None):
    with _lock:
        _active.pop(proc.pid if proc is not None else getattr(_local, "pid", None), None)


def active(owner: str) -> bool:
    with _lock:
        return any(o == owner for _, o in _active.values())


def cancel(owner=None, grace_s=1.0):
    with _lock:
        if owner is not None:
            _cancelled.add(owner)
        procs = [p for p, o in _active.values() if owner is None or o == owner]
    count = 0
    for proc in procs:
        if proc.poll() is None or getattr(proc, "_coresmith_process_scope", ""):
            _reap_process_group(proc, proc.pid, grace_s=grace_s)
            count += 1
        unregister(proc)
    return count


def _killpg_safe(pgid: int, sig: int) -> None:
    """os.killpg swallowing the benign 'group already gone' errors."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


_PROCESS_SCOPE_ENV = "CORESMITH_CALL_PROCESS_SCOPE"


def _signal_scoped_processes(scope: str, sig: int) -> int:
    """Signal Linux descendants that escaped the CLI's process group.

    Tool runners may start a new session, then orphan a scratch simulator.
    A unique per-call inherited environment marker preserves ownership after
    reparenting. Never match by command, working directory, or user alone.
    pidfds pin the inspected process so PID reuse cannot target another job.
    Environments are compared in memory and never logged.
    """
    if not isinstance(scope, str) or not scope or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return 0
    needle = f"{_PROCESS_SCOPE_ENV}={scope}".encode()
    ancestor_prefix = b"CORESMITH_PROCESS_ANCESTORS="
    count = 0
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        descriptor = None
        try:
            descriptor = os.pidfd_open(int(entry.name), 0)
            entries_env = (entry / "environ").read_bytes().split(b"\0")
            ancestor_match = any(v.startswith(ancestor_prefix) and scope.encode() in v[len(ancestor_prefix):].split(b":") for v in entries_env)
            if needle in entries_env or ancestor_match:
                signal.pidfd_send_signal(descriptor, sig)
                count += 1
        except (OSError, ValueError):
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return count


def _reap_process_group(
    process: subprocess.Popen, pgid: int, grace_s: float = 10.0
) -> None:
    """Terminate the child's process group so no grandchild survives the call.

    The CLI is launched with ``start_new_session=True``, so the child is its
    own session/group leader and ``pgid == child.pid`` at spawn. A grandchild
    the CLI spawned (e.g. a sim process) shares that pgid and inherits our
    stdout/stderr write-end; if it lives on after the CLI's final response, the
    reader threads block on the still-open pipe until the hard-timeout deadline
    (the observed ~45-min post-response exit stall).

    We SIGTERM the group (releasing grandchildren gracefully), reap the direct
    child within ``grace_s``, then SIGKILL any group survivors. Capturing
    ``pgid`` at spawn (rather than re-deriving via ``os.getpgid`` after the
    child may already be reaped) avoids the pid-reuse race.
    """
    scope = getattr(process, "_coresmith_process_scope", "")
    _signal_scoped_processes(scope, signal.SIGTERM)
    # Graceful: let the whole group wind down. If only the (already-exited)
    # leader remains, this is a no-op ESRCH.
    _killpg_safe(pgid, signal.SIGTERM)
    # Make sure the direct child is reaped (poll() may already have done this;
    # wait() then returns immediately with the cached returncode).
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        pass
    # Hard-kill any grandchild that ignored SIGTERM and is still holding pipes.
    _killpg_safe(pgid, signal.SIGKILL)
    _signal_scoped_processes(scope, signal.SIGKILL)



def popen(*args, **kwargs):
    with _lock:
        if _owner.get() in _cancelled:
            raise RuntimeError("job cancelled; refusing a new child")
        env = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        inherited = env.get(_PROCESS_SCOPE_ENV)
        if inherited:
            env["CORESMITH_PROCESS_ANCESTORS"] = ":".join(filter(None, [env.get("CORESMITH_PROCESS_ANCESTORS", ""), inherited]))
        env[_PROCESS_SCOPE_ENV] = uuid.uuid4().hex
        kwargs.update(env=env, start_new_session=True)
        proc = subprocess.Popen(*args, **kwargs)
        proc._coresmith_process_scope = env[_PROCESS_SCOPE_ENV]
        register(proc)
        return proc


@contextmanager
def output_lock(directory):
    if directory is None:
        yield
        return
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".coresmith-job.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"output directory already has an active job: {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(args, *, input=None, capture_output=False, timeout=None, check=False,
        output_dir=None, **kwargs):
    """subprocess.run semantics, plus whole-tree cleanup and output ownership."""
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input are mutually exclusive")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("capture_output conflicts with stdout/stderr")
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.monotonic()
    record = {"status": "unavailable", "exit_code": None, "elapsed_s": 0,
              "owner": _owner.get(), "executable": str(args[0]), "timeout_s": timeout}
    proc = None
    try:
        with output_lock(output_dir):
            proc = popen(args, **kwargs)
            try:
                stdout, stderr = proc.communicate(input, timeout=timeout)
                record.update(exit_code=proc.returncode,
                              status="passed" if proc.returncode == 0 else "failed")
                result = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
                result.execution = record
                if check:
                    result.check_returncode()
                return result
            except subprocess.TimeoutExpired as exc:
                record["status"] = "timed_out"
                _reap_process_group(proc, proc.pid, grace_s=1)
                exc.stdout, exc.stderr = proc.communicate()
                record["exit_code"] = proc.returncode
                exc.execution = record
                raise
            finally:
                _reap_process_group(proc, proc.pid, grace_s=1)
                unregister(proc)
    finally:
        record["elapsed_s"] = round(time.monotonic() - started, 3)
        if _owner.get() in _cancelled:
            record["status"] = "cancelled"
        root = os.environ.get("CORESMITH_PROJECT_ROOT")
        if root:
            folder = Path(root) / ".coresmith" / "jobs"
            try:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / (uuid.uuid4().hex + ".json")).write_text(json.dumps(record, indent=2))
            except OSError:
                pass
