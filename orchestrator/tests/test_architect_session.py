# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Architect sitting step 3: the session runner against a fake `claude` binary."""
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.architect import ArchitectSession, build_system_prompt
from orchestrator.state_store.project_db import open_project
from orchestrator.tests.test_ontology_stages import _PRD, _write

ROOT = Path(__file__).resolve().parents[2]

# A fake claude: emits stream-json (system init with a session id, one assistant
# text, a result), records its argv, and runs $FAKE_CLAUDE_DO (a shell snippet)
# in the cwd so a test can make "the model" do work through the CLI.
_FAKE = """#!/usr/bin/env bash
set -e
prompt=$(cat)
echo "$@" >> "$FAKE_LOG"
printf '%s\\n' "$prompt" > "$FAKE_LOG.prompt.$(date +%s%N)"
sid="sess-1"
for a in "$@"; do [ -n "$seen" ] && sid="$a" && break; [ "$a" = "--resume" ] && seen=1; done
echo '{"type":"system","subtype":"init","session_id":"'"$sid"'"}'
if [ -n "$FAKE_CLAUDE_DO" ]; then bash -c "$FAKE_CLAUDE_DO" >&2 || true; fi
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"did some work"}]}}'
echo '{"type":"result","subtype":"success","session_id":"'"$sid"'","total_cost_usd":0.5,"result":"ok"}'
"""


def _fake_claude(tmp_path) -> str:
    p = tmp_path / "fake_claude.sh"
    p.write_text(_FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_system_prompt_is_the_contract_plus_reference_appendix():
    s = build_system_prompt()
    assert "coresmith stage next" in s and "Nothing exists until it is registered" in s
    assert "## uarch_spec_generator.md" in s and "## skills/soc_fabric.md" in s
    assert "{prd_context}" not in s and "<prd_context>" in s
    assert len(s) > 100_000


def test_runner_stops_at_max_sittings_and_resumes_the_same_session(tmp_path, monkeypatch):
    _write(tmp_path)
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAKE_CLAUDE_DO", raising=False)
    sess = ArchitectSession(tmp_path, claude_path=_fake_claude(tmp_path), max_sittings=3, max_turns=7,
                            coresmith_bin=str(ROOT / "bin" / "coresmith"))
    st = sess.run()
    assert st["state"] == "blocked" and st["sittings"] == 3 and st["stage"] == "requirements"
    assert st["cost_usd"] == pytest.approx(1.5) and st["session_id"] == "sess-1"
    calls = log.read_text().splitlines()
    assert "--resume" not in calls[0] and "--resume sess-1" in calls[1] and "--resume sess-1" in calls[2]
    assert "--max-turns 7" in calls[0] and "--permission-mode auto" in calls[0]
    prompts = sorted(tmp_path.glob(".coresmith/architect/prompt-*.md"))
    assert len(prompts) == 3 and "Sitting 2" in prompts[1].read_text() and "MISSING_ARTIFACT" in prompts[1].read_text()
    assert (tmp_path / ".coresmith" / "architect" / "transcript-1.jsonl").exists()
    db = open_project(tmp_path)
    assert db.get_setting("architect_session_id") == "sess-1" and db.get_setting("architect_state") == "blocked"


def test_runner_finishes_when_the_model_drives_the_stages_through_the_cli(tmp_path, monkeypatch):
    _write(tmp_path)
    prd = dict(_PRD)
    prd["prd"] = dict(_PRD["prd"], functional_requirements=[f for f in _PRD["prd"]["functional_requirements"] if ":" in f],
                      open_items=[])
    (tmp_path / ".coresmith" / "prd_spec.json").write_text(json.dumps(prd))
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    cs = str(ROOT / "bin" / "coresmith")
    # "the model": one stage per sitting, exactly like the contract says
    script = tmp_path / "do.sh"
    script.write_text(f"""
set -e
S=$({cs} stage status --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["stage"])')
case "$S" in
  requirements) {cs} register prd .coresmith/prd_spec.json >/dev/null; {cs} register frd arch/frd_spec.md >/dev/null; {cs} stage next >/dev/null ;;
  arch_model) mkdir -p model/arch; echo '{{}}' > model/arch/arch_model.json; {cs} model register --arch >/dev/null
              for i in PERF-001 INV-001; do {cs} check add $i model_eval pass --evidence ok >/dev/null; done
              {cs} check add PHYS-001 model_eval not_testable --evidence physical >/dev/null; {cs} stage next >/dev/null ;;
  decomposition) {cs} register block_diagram .coresmith/block_diagram.json >/dev/null; {cs} register ers .coresmith/ers_spec.json >/dev/null; {cs} stage next >/dev/null ;;
  interfaces) python3 - <<'EOF'
import json,os
from orchestrator.state_store.project_db import open_project
db=open_project(os.environ['CORESMITH_PROJECT_ROOT'])
db.register_artifact('contracts','x',sha='a'); db.register_artifact('abi','x',sha='b')
open(os.path.join(os.environ['CORESMITH_PROJECT_ROOT'],'.coresmith','vip_index.json'),'w').write('{{}}')
db.add_integration_snapshot({{'tier': 'init', 'top': 'soc_top', 'rtl_path': '', 'real_blocks': [], 'stub_blocks': ['core'], 'wires': 1, 'boundary_ports': 5, 'elaborated': True, 'smoke_ok': None, 'wiring_errors': [], 'elab_errors': [], 'edge_results': {{}}}})
EOF
              {cs} stage next >/dev/null ;;
  uarch) for b in core l1d bus; do printf '## 2\\n## 3\\n### 4a\\n## 5\\n### 6a MEETS PERF-001\\n## 9\\n' > arch/u_$b.md; {cs} register uarch --block $b arch/u_$b.md >/dev/null; done; {cs} stage next >/dev/null ;;
  model_eval) python3 - <<'EOF'
import json,os
from orchestrator.state_store.project_db import open_project
db=open_project(os.environ['CORESMITH_PROJECT_ROOT'])
for b in ('core','l1d','bus','fabric'): db.upsert_model(b, path='m', sha='s', spec_contract_version='1', build_ok=True, smoke_ok=True)
open(os.path.join(os.environ['CORESMITH_PROJECT_ROOT'],'.coresmith','frd_eval.json'),'w').write(json.dumps({{"summary": {{"gate_ok": True}}}}))
EOF
              {cs} stage next >/dev/null ;;
esac
""")
    monkeypatch.setenv("FAKE_CLAUDE_DO", f"PYTHONPATH={ROOT} bash {script}")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    sess = ArchitectSession(tmp_path, claude_path=_fake_claude(tmp_path), max_sittings=8, coresmith_bin=cs)
    st = sess.run()
    assert st["state"] == "done", st
    assert st["stage"] == "blocks" and st["sittings"] == 6
    db = open_project(tmp_path)
    assert [r["name"] for r in db.stage_rows() if r["status"] == "done"] == \
        ["requirements", "arch_model", "decomposition", "interfaces", "uarch", "model_eval"]


def test_stop_file_and_cli_status(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    monkeypatch.delenv("FAKE_CLAUDE_DO", raising=False)
    sess = ArchitectSession(tmp_path, claude_path=_fake_claude(tmp_path), max_sittings=3, coresmith_bin=str(ROOT / "bin" / "coresmith"))
    (sess.dir / "STOP").write_text("1")
    assert sess.run()["state"] == "stopped"
    env = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, str(ROOT / "bin" / "coresmith"), "architect", "status", "--json"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and json.loads(p.stdout)["state"] == "stopped"
