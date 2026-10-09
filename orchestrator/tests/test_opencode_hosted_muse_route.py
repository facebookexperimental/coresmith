# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The Meta Model API provider block (and META_MODEL_API_KEY) is needed ONLY
for a ``meta-model-api/...`` model. A Muse model on a hosted route
(``opencode/muse-spark-...``, ``openrouter/...``) is passed through untouched:
the engine's own opencode calls (model eval/refine, integration review, cluster
workers) must not demand the Meta key the Architect does not need.

Driven end to end through ``ClaudeLLM`` against a fake ``opencode`` binary that
records its argv and environment."""
from __future__ import annotations

import json
import stat

import pytest

from orchestrator.langchain.agents import coresmith_llm as llm
from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM

_FAKE = r"""#!/usr/bin/env bash
cat > /dev/null
printf '%s\n' "$*" > "$FAKE_LOG.argv"
printf '%s' "${OPENCODE_CONFIG_CONTENT-<unset>}" > "$FAKE_LOG.config"
printf '%s' "${META_MODEL_API_KEY-<unset>}" > "$FAKE_LOG.key"
echo '{"type":"step_start","sessionID":"ses_f","part":{"type":"step-start"}}'
echo '{"type":"text","sessionID":"ses_f","part":{"type":"text","text":"ready"}}'
echo '{"type":"step_finish","sessionID":"ses_f","part":{"type":"step-finish","reason":"stop","tokens":{"total":2,"input":1,"output":1,"reasoning":0,"cache":{"read":0,"write":0}},"cost":0}}'
"""

_ENV = ("CORESMITH_OPENCODE_ENDPOINT", "CORESMITH_OPENCODE_MODEL", "CORESMITH_MODEL", "CORESMITH_BLOCK_MODEL",
        "META_MODEL_API_KEY", "OPENCODE_CONFIG_CONTENT", "CORESMITH_OPENCODE_VARIANT",
        "CORESMITH_OPENCODE_MAX_RETRIES")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / "opencode"
    p.write_text(_FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "oc"
    monkeypatch.setenv("FAKE_LOG", str(log))
    # after conftest's no-live-LLM stub: this test's fake IS the binary
    monkeypatch.setattr(llm, "_find_opencode_binary", lambda *_a, **_k: str(p))
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "opencode")
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CORESMITH_OPENCODE_MAX_RETRIES", "0")
    return log


def _argv_model(log) -> str:
    argv = (log.parent / (log.name + ".argv")).read_text().split()
    return argv[argv.index("--model") + 1]


def test_hosted_opencode_muse_model_needs_no_meta_provider_or_key(fake, monkeypatch):
    monkeypatch.setenv("CORESMITH_OPENCODE_MODEL", "opencode/muse-spark-1.3-contributor-free")
    out = ClaudeLLM(model="opus-5", timeout=30)._generate_via_cli("system", "hello")
    assert "ready" in out
    assert _argv_model(fake) == "opencode/muse-spark-1.3-contributor-free"
    cfg = (fake.parent / (fake.name + ".config")).read_text()
    assert llm.MUSE_SPARK_PROVIDER_ID not in cfg            # no provider injection
    assert (fake.parent / (fake.name + ".key")).read_text() == "<unset>"


@pytest.mark.parametrize("model", ["openrouter/meta/muse-spark-1.1", "opencode/muse-spark-1.3-contributor-free"])
def test_explicit_non_meta_route_is_passed_through_even_on_the_muse_endpoint(fake, monkeypatch, model):
    monkeypatch.setenv("CORESMITH_OPENCODE_ENDPOINT", "muse")
    monkeypatch.setenv("CORESMITH_OPENCODE_MODEL", model)
    ClaudeLLM(model="opus-5", timeout=30)._generate_via_cli("system", "hello")
    assert _argv_model(fake) == model
    assert llm.MUSE_SPARK_PROVIDER_ID not in (fake.parent / (fake.name + ".config")).read_text()


def test_muse_endpoint_without_a_model_still_demands_the_key(fake, monkeypatch):
    monkeypatch.setenv("CORESMITH_OPENCODE_ENDPOINT", "muse")
    with pytest.raises(RuntimeError, match="META_MODEL_API_KEY is not set"):
        ClaudeLLM(model="opus-5", timeout=30)._generate_via_cli("system", "hello")
    assert not (fake.parent / (fake.name + ".argv")).exists()  # never launched


def test_muse_endpoint_with_the_key_registers_the_provider(fake, monkeypatch):
    monkeypatch.setenv("CORESMITH_OPENCODE_ENDPOINT", "muse")
    monkeypatch.setenv("META_MODEL_API_KEY", "LLM_k")
    ClaudeLLM(model="opus-5", timeout=30)._generate_via_cli("system", "hello")
    assert _argv_model(fake) == llm.DEFAULT_MUSE_SPARK_MODEL
    cfg = json.loads((fake.parent / (fake.name + ".config")).read_text())
    assert llm.MUSE_SPARK_PROVIDER_ID in cfg["provider"]


@pytest.mark.parametrize("slug,needs", [
    ("meta-model-api/muse-spark-1.3-contributor", True),
    ("META-MODEL-API/muse-spark-1.1", True),
    ("opencode/muse-spark-1.3-contributor-free", False),
    ("openrouter/meta/muse-spark-1.1", False),
    ("openrouter/moonshotai/kimi-k3", False),
    ("", False),
])
def test_needs_meta_model_api_provider(slug, needs):
    assert llm._needs_meta_model_api_provider(slug) is needs
