# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Reference-entrypoint resolution must pick the oracle by INTENT, not alphabet.

Regression: a codec golden exposing both ``encode`` and ``decode`` resolved to
``decode`` for an ENCODER design, because discovery returned the first ``dir()``
(alphabetical) match and 'd' < 'e'. The composition gate then had the wrong
oracle and raised ``reference_uninvokable``. These pin the intent ranking.
"""
from __future__ import annotations

import types

import pytest

from orchestrator.architecture import composition


def _mk_module(name: str, func_names: list[str]) -> types.ModuleType:
    """A fake reference module with real public top-level functions."""
    mod = types.ModuleType(name)
    mod.__name__ = name
    for fn_name in func_names:
        def _f(*a, **k):  # noqa: ANN002, ANN003 - toy
            return None
        _f.__name__ = fn_name
        _f.__module__ = name
        setattr(mod, fn_name, _f)
    return mod


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)


def test_encoder_golden_with_both_prefers_encode(tmp_path):
    """THE BUG: a golden with both encode+decode must resolve to encode."""
    mod = _mk_module("toy_codec", ["decode", "encode", "aux_helper"])
    fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
    assert name == "encode", "must pick encode by intent, not alphabetical decode"
    assert fn is mod.encode


def test_decoder_only_golden_picks_decode(tmp_path):
    """A pure decoder (only decode present) still resolves to decode."""
    mod = _mk_module("toy_dec", ["decode", "helper_thing"])
    fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
    assert name == "decode"
    assert fn is mod.decode


def test_env_override_still_wins(tmp_path, monkeypatch):
    """An explicit CORESMITH_REFERENCE_ENTRY overrides discovery entirely."""
    monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "decode")
    mod = _mk_module("toy_codec", ["decode", "encode"])
    fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
    assert name == "decode" and fn is mod.decode


def test_declared_entry_in_prd_wins_over_discovery(tmp_path):
    """A declared reference_entry_point in prd_spec.md beats discovery."""
    arch = tmp_path / "arch"
    arch.mkdir()
    (arch / "prd_spec.md").write_text("reference_entry_point: decode\n")
    mod = _mk_module("toy_codec", ["decode", "encode"])
    fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
    assert name == "decode"


def test_entry_priority_orders_encode_before_decode():
    assert composition._entry_priority("encode") < composition._entry_priority("decode")
    # encode_image variant ranks as an encode (not after decode)
    assert composition._entry_priority("encode_image_frame") < composition._entry_priority("decode")
    # generic drivers rank after the primary transforms
    assert composition._entry_priority("encode") < composition._entry_priority("run")
    assert composition._entry_priority("decode") < composition._entry_priority("main")


def test_single_public_fn_still_used_when_no_conventional_name(tmp_path):
    """Fallback: exactly one public fn with a non-conventional name is used."""
    mod = _mk_module("toy_thing", ["transmogrify"])
    fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
    assert name == "transmogrify"
