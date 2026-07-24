# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The MyHDL sim can run in a subprocess under CORESMITH_SIM_PYTHON (e.g. pypy3).

Validates the mechanism with CPython-as-the-interpreter: the subprocess path
must return results IDENTICAL to the in-process thread path, honour the hard
timeout (and actually kill the child), surface sim errors, and the dispatcher
must default to in-process when the knob is unset.
"""
from __future__ import annotations

import sys

from orchestrator.architecture import model_integration as mi

# A tiny "chip model": simulate(stimulus) -> (output, cycles). Pure Python.
_CHIP = '''
def simulate(stimulus):
    n = int(stimulus["n"])
    acc = 0
    for i in range(n):
        acc = (acc + i * 3) & 0xffff
    return ([acc & 0xff, (acc >> 8) & 0xff], n)
'''

_SLOW_CHIP = '''
import time
def simulate(stimulus):
    time.sleep(30)
    return ([0], 1)
'''

_BROKEN_CHIP = '''
def simulate(stimulus):
    raise ValueError("boom in sim")
'''


def _write_model(tmp_path, src):
    models = tmp_path / "arch" / "block_models"
    models.mkdir(parents=True)
    path = models / "_chip_model.py"
    path.write_text(src)
    return path, models


def test_subprocess_matches_inprocess(tmp_path):
    path, models = _write_model(tmp_path, _CHIP)
    ns: dict = {}
    exec(_CHIP, ns)
    stim = {"n": 5000}
    inproc, t1, e1 = mi._run_simulate_with_timeout(ns["simulate"], stim, 30)
    sub, t2, e2 = mi._run_simulate_subprocess(path, models, stim, 30, sys.executable)
    assert (t1, e1, t2, e2) == (False, None, False, None)
    assert sub == inproc, "subprocess sim must be bit-identical to in-process"


def test_subprocess_timeout_kills_child(tmp_path):
    path, models = _write_model(tmp_path, _SLOW_CHIP)
    res, timed_out, exc = mi._run_simulate_subprocess(
        path, models, {"n": 1}, timeout_s=1.0, interpreter=sys.executable
    )
    assert timed_out is True and res is None


def test_subprocess_reports_sim_error(tmp_path):
    path, models = _write_model(tmp_path, _BROKEN_CHIP)
    res, timed_out, exc = mi._run_simulate_subprocess(
        path, models, {"n": 1}, 30, sys.executable
    )
    assert not timed_out and res is None
    assert exc is not None and "boom in sim" in str(exc)


def test_subprocess_missing_interpreter(tmp_path):
    path, models = _write_model(tmp_path, _CHIP)
    res, timed_out, exc = mi._run_simulate_subprocess(
        path, models, {"n": 1}, 30, "pypy3-does-not-exist-xyz"
    )
    assert res is None and not timed_out
    assert "not found" in str(exc)


def test_dispatch_defaults_to_inprocess(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_SIM_PYTHON", raising=False)
    path, models = _write_model(tmp_path, _CHIP)
    ns: dict = {}
    exec(_CHIP, ns)
    # with the knob unset, dispatch must use the in-process path (works even if
    # chip_model_path were bogus, because the subprocess is never spawned)
    res, t, e = mi._dispatch_simulate(ns["simulate"], "/nonexistent", models,
                                      {"n": 100}, 30)
    assert t is False and e is None and res == ns["simulate"]({"n": 100})


def test_dispatch_uses_subprocess_when_knob_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_SIM_PYTHON", sys.executable)
    path, models = _write_model(tmp_path, _CHIP)
    # in-process callable is intentionally WRONG to prove the subprocess (which
    # re-imports the real chip model from disk) is what actually runs
    def wrong(s):
        return [999], -1
    res, t, e = mi._dispatch_simulate(wrong, path, models, {"n": 5000}, 30)
    ns: dict = {}
    exec(_CHIP, ns)
    assert res == ns["simulate"]({"n": 5000}) and res != wrong({"n": 5000})
