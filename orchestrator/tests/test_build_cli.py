# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith build ...`` is machine-readable on every path: with ``--json``
a refusal, an invalid input and an unreachable daemon are JSON documents on
stdout with the harness exit codes (2 refused / invalid, 3 infrastructure),
and the read-only ledger verbs report an unknown build the same way."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from orchestrator.tests.build_fixtures import complete_build, ready_project

_REPO = Path(__file__).resolve().parents[2]


def _run(root, *argv):
    env = {"CORESMITH_PROJECT_ROOT": str(root), "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "PYTHONPATH": str(_REPO), "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run([sys.executable, str(_REPO / "bin" / "coresmith"), *argv],
                          capture_output=True, text=True, env=env, timeout=180)


def test_build_verbs_report_an_unreachable_daemon_as_json_with_exit_3(tmp_path, monkeypatch):
    ready_project(tmp_path, monkeypatch)
    for argv in (("build", "module", "tiny", "--json"), ("build", "status", "--json"),
                 ("build", "resume", "--build-id", "b-x", "--action", "approve", "--json"),
                 ("build", "abort", "--build-id", "b-x", "--json"), ("build", "pause", "--json")):
        p = _run(tmp_path, *argv)
        assert p.returncode == 3, (argv, p.stdout, p.stderr)
        doc = json.loads(p.stdout)
        assert doc["error"] == "DAEMON_UNAVAILABLE" and "daemon" in doc["message"]
    # without --json: a stderr line, the same exit code, nothing on stdout
    p = _run(tmp_path, "build", "module", "tiny")
    assert p.returncode == 3 and p.stdout == "" and "DAEMON_UNAVAILABLE" in p.stderr


def test_a_daemon_answering_non_json_is_an_infrastructure_error_even_on_200(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Text(BaseHTTPRequestHandler):
        def _answer(self):
            body = b"<html>not the daemon</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        do_GET = do_POST = _answer

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), _Text)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ready_project(tmp_path, monkeypatch)
        (tmp_path / ".coresmith" / "daemon.json").write_text(json.dumps({"port": srv.server_address[1], "pid": os.getpid()}))
        for argv in (("build", "status", "--json"), ("build", "module", "tiny", "--json")):
            p = _run(tmp_path, *argv)
            assert p.returncode == 3, (argv, p.stdout, p.stderr)
            doc = json.loads(p.stdout)
            assert doc["error"] == "DAEMON_BAD_RESPONSE" and doc["http_status"] == 200
    finally:
        srv.shutdown()


def test_ledger_verbs_report_an_unknown_build_as_json_with_exit_2(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    bid = complete_build(db, tmp_path, "tiny")
    p = _run(tmp_path, "build", "show", "b-nope", "--json")
    assert p.returncode == 2 and json.loads(p.stdout) == {"error": "UNKNOWN_BUILD", "message": "no build b-nope"}
    p = _run(tmp_path, "build", "compare", bid, "b-nope", "--json")
    assert p.returncode == 2 and json.loads(p.stdout)["error"] == "UNKNOWN_BUILD"
    p = _run(tmp_path, "build", "show", "b-nope")
    assert p.returncode == 2 and p.stdout == "" and "UNKNOWN_BUILD" in p.stderr
    p = _run(tmp_path, "build", "show", bid, "--json")
    assert p.returncode == 0, p.stderr
    doc = json.loads(p.stdout)
    assert doc["build"]["id"] == bid and doc["current"]["ok"] is True and doc["evidence"]["ppa"]
    p = _run(tmp_path, "build", "lineage", "--json")
    assert p.returncode == 0 and json.loads(p.stdout)["modules"]["tiny"]["published"]["build_id"] == bid
    p = _run(tmp_path, "build", "list", "--json")
    assert p.returncode == 0 and [b["id"] for b in json.loads(p.stdout)["builds"]] == [bid]


def test_removed_build_options_are_rejected(tmp_path):
    import pytest
    from pydantic import ValidationError

    from orchestrator.daemon.server import BuildModuleRequest

    for name, value in (("objective", "AREA-CPU-1:min"), ("improve_rounds", 2)):
        flag = "--" + name.replace("_", "-")
        for verb in ("module", "targets"):
            result = _run(tmp_path, "build", verb, "tiny", flag, str(value), "--json")
            assert result.returncode == 2
            assert "unrecognized arguments" in result.stdout + result.stderr
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BuildModuleRequest(module="tiny", **{name: value})
