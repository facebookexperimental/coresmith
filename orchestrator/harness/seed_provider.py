# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unified DV/equivalence seed provider -- the dev/gate train-test boundary.

Every deterministic anti-memorization check in CoreSmith drives BOTH the RTL and
a reference on the SAME seeded-random stimulus and byte-compares. The seed is the
whole anti-cheat: it is drawn fresh *after* the RTL is frozen, so a finite
stimulus-keyed LUT cannot have memorised it -- only a genuine datapath matches.

Historically the ``CORESMITH_DV_SEED_PIN``-else-``secrets.randbits(63)`` draw was
re-implemented in five places (pipeline_helpers mint, harness/verify,
pipeline_graph equiv, model_integration gate, branch_parity). This module is the
single source of truth, and it makes explicit the distinction the
"oracles-as-tools" design turns on:

* ``gate_seed()`` -- the AUTHORITATIVE seed for a gate run. Fresh per call
  (post-freeze), honours ``CORESMITH_DV_SEED_PIN`` only for reproducible
  operator debugging. **Never surface a gate seed into LLM-visible context** --
  it is the un-memorisable exam. This is byte-identical to the five call sites it
  replaces (PIN else randbits), so gate behaviour is unchanged.

* ``dev_seed()`` -- the PRACTICE seed for the checker CLIs (``verify equiv`` /
  ``golden-check``). Takes an explicit ``--seed`` for reproducing a failing case,
  else PIN, else fresh random. It is meant to be PRINTED so a developer/agent can
  re-run the same case -- and precisely because it is visible, it must never be
  reused as a gate seed. Passing on N random dev seeds approximates correctness;
  the gate is one more draw the practice never saw.
"""
from __future__ import annotations

import os
import secrets

DV_SEED_ENV = "CORESMITH_DV_SEED"
DV_SEED_PIN_ENV = "CORESMITH_DV_SEED_PIN"


def _pinned_seed() -> int | None:
    """The operator-pinned seed (reproducible debugging), or None if unset/bad."""
    pin = (os.environ.get(DV_SEED_PIN_ENV) or "").strip()
    if not pin:
        return None
    try:
        return int(pin)
    except ValueError:
        return None


def gate_seed(explicit: int | None = None, use_env: bool = False) -> int:
    """Authoritative gate seed: fresh cryptographic 63-bit value.

    Resolution order: an ``explicit`` caller value (rare -- e.g. re-running a
    specific failure) > (when ``use_env``) an already-minted ``CORESMITH_DV_SEED``
    from the pipeline mint point > ``CORESMITH_DV_SEED_PIN`` (operator debug) >
    a fresh ``secrets.randbits(63)``. ``use_env=True`` is for gates that consume
    the seed the block-DV mint already injected into the sim env so both see the
    same stimulus in one run (harness/verify's historical behaviour).
    """
    if explicit is not None:
        return int(explicit)
    if use_env:
        env_val = (os.environ.get(DV_SEED_ENV) or "").strip()
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass
    pin = _pinned_seed()
    if pin is not None:
        return pin
    return secrets.randbits(63)


def mint_dv_seed(env: dict) -> str:
    """Mint the per-run block-DV seed and inject it into ``env`` as a string.

    The pipeline mint point: sets ``env[CORESMITH_DV_SEED]`` to the pinned value
    (reproducible debug) else a fresh 63-bit seed, and returns it. The cocotb TB
    reads ``os.environ[CORESMITH_DV_SEED]`` (testbench_generator rule 16).
    """
    pin = _pinned_seed()
    seed = str(pin) if pin is not None else str(secrets.randbits(63))
    env[DV_SEED_ENV] = seed
    return seed


def dev_seed(explicit: int | None = None) -> int:
    """Practice seed for the checker CLIs -- meant to be PRINTED, not a gate seed.

    ``explicit`` (a ``--seed`` value) reproduces a specific case; else PIN; else a
    fresh random. Deliberately independent of ``CORESMITH_DV_SEED`` env so a dev
    run never silently inherits (or contaminates) a gate's stimulus.
    """
    if explicit is not None:
        return int(explicit)
    pin = _pinned_seed()
    if pin is not None:
        return pin
    return secrets.randbits(63)
