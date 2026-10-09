"""Evidence joins, deduplication and privacy of the run viewer exporter.

Every fixture is synthetic and built in ``tmp_path``; nothing from a real run
is checked in.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.run_viewer import cli_calls, engine, export, lineage, native  # noqa: E402
from tools.run_viewer.privacy import (  # noqa: E402
    ENCRYPTED_RE,
    Ledger,
    mask_env_argv,
    redact,
    scrub,
)
from tools.run_viewer.sources import Sources  # noqa: E402

SIG = "EqQBCkYIBhgCKkA" + "Zm9vYmFyYmF6cXV4" * 12          # a signature-shaped base64 payload
ENC = "gAAAAABq" + "x0X93F-NJJogYb8S7DheYeiB5DGN" * 6      # a Fernet-shaped encrypted payload
THINK = "SYNTHETIC PRIVATE THINKING: compare the AXI handshake against the contract before answering"


def _turns_text(agent) -> str:
    return json.dumps(agent.turns, ensure_ascii=False)


# ------------------------------------------------------------------ privacy
def test_claude_thinking_and_signature_never_exported():
    led = Ledger()
    a = native.Agent("t", "a", kind="architect", label="A", provider="claude")
    rec = {"type": "assistant", "uuid": "u1", "timestamp": "2026-10-08T10:00:00Z",
           "message": {"id": "m1", "model": "claude-x", "content": [
               {"type": "thinking", "thinking": THINK, "signature": SIG},
               {"type": "redacted_thinking", "data": SIG},
               {"type": "text", "text": "Public answer."}]}}
    native.claude_record(a, rec, {"s": "S1", "l": 1}, led)
    led.resolve()
    blob = _turns_text(a)
    assert THINK not in blob and SIG not in blob
    assert "Public answer." in blob
    kinds = [t["kind"] for t in a.turns]
    assert kinds.count("thinking") == 2
    assert led.leaks(blob) == []
    assert led.leaks("leaked: " + THINK) and led.leaks(json.dumps({"x": SIG}))


def test_stream_partial_deltas_are_dropped_and_fingerprinted(tmp_path):
    led = Ledger()
    a = native.Agent("t", "architect", kind="architect", label="A", provider="claude")
    lines = [
        {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m9"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "thinking_delta", "thinking": THINK[:50]}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "thinking_delta", "thinking": THINK[50:]}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "signature_delta", "signature": SIG}}},
        {"type": "assistant", "uuid": "u9", "timestamp": "2026-10-08T10:00:01Z",
         "message": {"id": "m9", "content": [{"type": "text", "text": "visible"}]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    native.parse_claude_stream(a, p, Sources(tmp_path), led, "stream")
    led.resolve()
    blob = _turns_text(a)
    assert THINK[:40] not in blob and SIG[:40] not in blob
    assert led.leaks("x " + THINK + " y"), "the aggregated streamed thinking text is fingerprinted"
    assert led.leaks(SIG[100:220]), "a fragment from the middle of a signature is detected"


def test_narration_thinking_fully_visible_is_not_a_false_leak():
    led = Ledger()
    a = native.Agent("t", "a", kind="architect", label="A", provider="claude")
    narr = "Now checking the integration stage before run start, so the testbench is kept."
    native.claude_record(a, {"type": "assistant", "uuid": "n1", "message": {"id": "m1", "content": [
        {"type": "thinking", "thinking": narr, "signature": SIG}]}}, {"s": "S1", "l": 1}, led)
    native.claude_record(a, {"type": "assistant", "uuid": "n2", "message": {"id": "m1", "content": [
        {"type": "text", "text": narr}]}}, {"s": "S1", "l": 2}, led)
    led.resolve()
    assert led.leaks(_turns_text(a)) == []
    assert led.counts["thinking_text_fully_visible_in_same_agent"] == 1
    # a partial overlap never exempts: a private text whose fragment is
    # visible elsewhere stays fingerprinted
    led2 = Ledger()
    b = native.Agent("t", "b", kind="architect", label="B", provider="claude")
    native.claude_record(b, {"type": "assistant", "uuid": "p1", "message": {"id": "m2", "content": [
        {"type": "thinking", "thinking": narr + " Private follow-up reasoning that was never shown."}]}},
        {"s": "S1", "l": 1}, led2)
    native.claude_record(b, {"type": "assistant", "uuid": "p2", "message": {"id": "m2", "content": [
        {"type": "text", "text": narr}]}}, {"s": "S1", "l": 2}, led2)
    led2.resolve()
    assert led2.leaks("Private follow-up reasoning that was never shown.")


def test_embedded_private_fields_in_tool_output_are_withheld_at_any_depth():
    led = Ledger()
    raw = {"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": THINK, "signature": SIG}, {"type": "text", "text": "kept text"}]}}
    depth0 = json.dumps(raw)
    depth1 = json.dumps({"partial_stdout": depth0})
    depth2 = json.dumps({"outer": depth1})
    for text in (depth0, depth1, depth2, "log line: " + depth1 + " trailing"):
        out = redact(text, led)
        assert THINK not in out and SIG not in out, text[:40]
        assert "kept text" in out
    cut = depth1[: depth1.index(SIG) + 120]        # the shell cut the output mid-signature
    out = redact(cut, led)
    assert SIG[:60] not in out
    cut2 = depth0[: depth0.index(THINK) + 30]      # cut inside the thinking value
    assert THINK[:30] not in redact(cut2, led)


def test_copied_signature_fragment_without_key_is_withheld_and_public_hex_kept():
    led = Ledger()
    led.private("claude_thinking:signature", SIG)
    led.resolve()
    hexdata = "GOLD = \"" + "0123456789abcdef" * 20 + "\""
    text = "ls output\n" + SIG[37:300] + "\nnext line\n" + hexdata
    out = led.scrub_private_runs(text)
    assert SIG[37:300] not in out and "[private signature/encrypted fragment not exported" in out
    assert hexdata in out and "next line" in out


def test_codex_encrypted_payloads_become_markers(tmp_path):
    led = Ledger()
    recs = [
        {"type": "session_meta", "timestamp": "2026-10-08T10:00:00Z",
         "payload": {"id": "th1", "cwd": "/work", "creator_account_id": "acct-secret", "creator_user_id": "user-1"}},
        {"type": "response_item", "timestamp": "2026-10-08T10:00:01Z",
         "payload": {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": ENC}},
        {"type": "event_msg", "timestamp": "2026-10-08T10:00:01Z",
         "payload": {"type": "item_completed", "item": {"type": "Reasoning", "id": "rs1", "raw_content": [], "summary_text": []}}},
        {"type": "response_item", "timestamp": "2026-10-08T10:00:02Z",
         "payload": {"type": "function_call", "name": "spawn_agent", "call_id": "c1",
                     "arguments": json.dumps({"task_name": "audit", "message": ENC})}},
        {"type": "response_item", "timestamp": "2026-10-08T10:00:03Z",
         "payload": {"type": "agent_message", "id": "am1", "author": "/root/x", "recipient": "/root",
                     "content": [{"type": "input_text", "text": "Payload:"},
                                 {"type": "encrypted_content", "encrypted_content": ENC}]}},
        {"type": "response_item", "timestamp": "2026-10-08T10:00:04Z",
         "payload": {"type": "message", "id": "m1", "role": "assistant", "channel": "analysis",
                     "content": [{"type": "output_text", "text": THINK}]}},
    ]
    p = tmp_path / "rollout.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    a = native.Agent("t", "a", kind="architect", label="A", provider="codex")
    native.parse_codex_rollout(a, p, Sources(tmp_path), led, "rollout")
    blob = _turns_text(a) + json.dumps(a.meta)
    assert not ENCRYPTED_RE.search(blob)
    assert THINK not in blob and "acct-secret" not in blob
    assert sum(1 for t in a.turns if t["kind"] == "thinking") == 2, "Reasoning item and response copy fold to one marker"
    assert "Payload:" in blob and "task_name" in blob


def _project_db(path: Path, *, actions=(), builds=(), decisions=(), interrupts=(), dv=(), cov=(), ppa=(), results=()):
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE actions (id INTEGER PRIMARY KEY, ts REAL NOT NULL, actor TEXT, argv_json TEXT NOT NULL, rc INTEGER, summary TEXT, run_id TEXT);
    CREATE TABLE builds (id TEXT PRIMARY KEY, module TEXT, run_id TEXT, entry TEXT, graph TEXT, thread_id TEXT, checkpoint_ns TEXT,
      status TEXT, requested_at REAL, started_at REAL, finished_at REAL, inputs_json TEXT, worker_json TEXT, seed_json TEXT, result_json TEXT, error TEXT);
    CREATE TABLE decisions (id INTEGER PRIMARY KEY, interrupt_id TEXT, interrupt_type TEXT, block TEXT, action TEXT, reasoning TEXT,
      decision_index INTEGER, run_id TEXT, ts REAL, actor TEXT);
    CREATE TABLE interrupts (id TEXT PRIMARY KEY, lg_interrupt_id TEXT, graph TEXT, branch TEXT, node TEXT, block TEXT, kind TEXT,
      payload_json TEXT, status TEXT, resolution_json TEXT, resolved_by TEXT, run_id TEXT, ts REAL, resolved_ts REAL, consumed_ts REAL);
    CREATE TABLE dv_results (id INTEGER PRIMARY KEY, ts REAL, block TEXT, scope TEXT, source TEXT, attempt INTEGER, passed INTEGER,
      skipped INTEGER, tests_passed INTEGER, tests_total INTEGER, build_id TEXT, run_id TEXT);
    CREATE TABLE coverage_results (id INTEGER PRIMARY KEY, ts REAL, block TEXT, scope TEXT, points_total INTEGER, points_hit INTEGER,
      pct REAL, uncovered TEXT, build_id TEXT, attempt INTEGER, run_id TEXT);
    CREATE TABLE ppa_history (id INTEGER PRIMARY KEY, ts REAL, block TEXT, attempt INTEGER, source TEXT, probe TEXT, cells INTEGER,
      wns_ns REAL, ppa_ok INTEGER, build_id TEXT, power_basis TEXT, run_id TEXT);
    CREATE TABLE results (block TEXT, kind TEXT, value_json TEXT, report_path TEXT, ts REAL, PRIMARY KEY (block, kind));
    CREATE TABLE stages (name TEXT PRIMARY KEY, ordinal INTEGER, status TEXT, entered_ts REAL, done_ts REAL, blocked_by_json TEXT, ts REAL);
    """)
    con.executemany("INSERT INTO actions VALUES (?,?,?,?,?,?,?)", actions)
    con.executemany("INSERT INTO builds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", builds)
    con.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)", decisions)
    con.executemany("INSERT INTO interrupts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", interrupts)
    con.executemany("INSERT INTO dv_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", dv)
    con.executemany("INSERT INTO coverage_results VALUES (?,?,?,?,?,?,?,?,?,?,?)", cov)
    con.executemany("INSERT INTO ppa_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", ppa)
    con.executemany("INSERT INTO results VALUES (?,?,?,?,?)", results)
    con.commit()
    con.close()


def test_operator_rationale_is_public_and_database_is_not_touched(tmp_path):
    db = tmp_path / "project.sqlite"
    _project_db(db, decisions=[(1, "int-0123456789abcdef", "integration_dv_failure", "", "fix_tb",
                                "DV_PROCESS_ERROR: the TB worker ran out of turns; no RTL defect.", 1, "run-1", 10.0, "architect")],
                interrupts=[("int-0123456789abcdef", "lg", "pipeline", None, "integration_dv_failure", "", "integration_dv_failure",
                             "{}", "consumed", json.dumps({"action": "fix_tb", "reasoning": "operator rationale text"}),
                             "architect", "run-1", 5.0, 10.0, 10.0)])
    led = Ledger()
    before = sorted(p.name for p in tmp_path.iterdir())
    proj = engine.load_project(db, Sources(tmp_path), "t", led)
    assert proj["decisions"][0]["reasoning"].startswith("DV_PROCESS_ERROR")
    assert proj["interrupts"][0]["resolution"]["reasoning"] == "operator rationale text"
    lin = lineage.build_lineage("t", proj, {}, [], [], [], [])
    assert lin["parks"][0]["decision"]["reasoning"].startswith("DV_PROCESS_ERROR")
    assert lin["parks"][0]["decision"]["label"] == "exact"
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "reading must not create -wal/-shm sidecars"


def test_credentials_are_omitted_or_redacted():
    led = Ledger()
    a = native.Agent("t", "a", kind="architect", label="A", provider="claude")
    native.claude_record(a, {"type": "attachment", "uuid": "c1", "attachment": {"type": "credential_org", "org": "secret-org"}},
                         {"s": "S1", "l": 1}, led)
    assert a.turns == [] and led.counts["omitted_attachment:credential_org"] == 1
    tok = "sk-ant-" + "A" * 40
    assert tok not in redact("export KEY=" + tok, led)
    argv = mask_env_argv(["--env", "ANTHROPIC_API_KEY=abc123", "--env", "CORESMITH_MODEL=opus"])
    assert "abc123" not in " ".join(argv) and "CORESMITH_MODEL=opus" in argv
    assert scrub({"rate": {"credits": {"balance": "56"}}}, led)["rate"]["credits"] == "[withheld: private field]"


# ------------------------------------------------------------------ deduplication
def test_claude_uuid_dedup_across_native_session_and_stream(tmp_path):
    led = Ledger()
    a = native.Agent("t", "architect", kind="architect", label="A", provider="claude")
    rec = {"type": "assistant", "uuid": "same-uuid", "sessionId": "S", "timestamp": "2026-10-08T10:00:00Z",
           "message": {"id": "m1", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                                "input": {"command": "coresmith stage status"}}]}}
    (tmp_path / "s.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "t.jsonl").write_text(json.dumps({**rec, "session_id": "S"}) + "\n")
    src = Sources(tmp_path)
    native.parse_claude_session(a, tmp_path / "s.jsonl", src, led, "native")
    native.parse_claude_stream(a, tmp_path / "t.jsonl", src, led, "stream")
    assert len(a.turns) == 1
    assert [r["s"] for r in a.turns[0]["refs"]] == ["S1", "S2"]
    assert a.turns[0]["text"] == "coresmith stage status"


def test_event_copies_across_rotated_files_keep_references(tmp_path):
    e1 = {"ts": 100.0, "pid": 7, "event": "graph_node_enter", "node": "Init Block", "block": "uart", "build_id": "b-uart-1"}
    e2 = {"ts": 101.0, "pid": 7, "event": "graph_node_exit", "node": "Block Done", "block": "uart", "build_id": "b-uart-1"}
    (tmp_path / "pipeline_events.20261008-120000.jsonl").write_text(json.dumps(e1) + "\n" + json.dumps(e2) + "\n")
    (tmp_path / "pipeline_events.jsonl").write_text(json.dumps(e2) + "\n" + json.dumps({**e1, "ts": 200.0}) + "\n")
    out = engine.load_events(tmp_path, Sources(tmp_path), "t", Ledger())
    evs = out["events"]
    assert len(evs) == 3
    done = [e for e in evs if e["node"] == "Block Done"]
    assert len(done) == 1 and len(done[0]["copies"]) == 1 and done[0]["active_file"] is False
    assert [f["active"] for f in out["files"]] == [False, True]


def test_codex_exec_items_reconcile_with_rollout_and_item_ids_are_scoped_per_call(tmp_path):
    led = Ledger()
    roll = [{"type": "event_msg", "timestamp": "2026-10-08T10:00:05Z",
             "payload": {"type": "item_completed", "completed_at_ms": 1791453605000,
                         "item": {"type": "CommandExecution", "id": "exec-1", "command": ["/bin/bash", "-lc", "make lint"],
                                  "aggregated_output": "lint ok\n", "exit_code": "0", "duration": {"secs": 2, "nanos": 0}}}}]
    p = tmp_path / "rollout.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in roll))
    src = Sources(tmp_path)
    a = native.Agent("t", "h1.01", kind="engine-helper", label="h", provider="codex")
    native.parse_codex_rollout(a, p, src, led, "rollout")
    sid = src.add(p, "turn log")
    exec_events = [(1, 1791453603.0, {"type": "item.started", "item": {"id": "item_0", "type": "command_execution"}}),
                   (2, 1791453605.0, {"type": "item.completed", "item": {"id": "item_0", "type": "command_execution",
                                                                         "command": "/bin/bash -lc 'make lint'",
                                                                         "aggregated_output": "lint ok\n", "exit_code": 0}}),
                   (3, 1791453606.0, {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                                                                         "text": "only in the exec stream"}})]
    prior = native.reconcile_codex_exec(a, exec_events, sid, src, led, call_scope="pid1:1.0")
    cmd_turns = [t for t in a.turns if t.get("name") == "exec_command"]
    assert len(cmd_turns) == 1 and len(cmd_turns[0]["refs"]) == 2
    only = [t for t in a.turns if "exec_stream_only" in t["flags"]]
    assert len(only) == 1 and only[0]["text"] == "only in the exec stream"
    # an identical copy of the same stream (live capture) only adds references
    native.reconcile_codex_exec(a, [(9, None, ev) for _, _, ev in exec_events], sid, src, led,
                                call_scope="pid1:1.0", prior=prior)
    assert len([t for t in a.turns if "exec_stream_only" in t["flags"]]) == 1
    # another call reusing item_0 is a different operation
    b = native.Agent("t", "h1.02", kind="engine-helper", label="h2", provider="codex")
    native.reconcile_codex_exec(b, exec_events, sid, src, led, call_scope="pid2:2.0")
    assert len([t for t in b.turns if t.get("name") == "exec_command"]) == 1


# ------------------------------------------------------------------ CLI joins
def test_cli_join_strong_loop_expansion_wrapper_and_unaudited_help():
    wrappers = cli_calls.find_wrappers([
        ("architect:5", "Bash", "cat > /work/cs.sh <<'EOF'\n#!/bin/bash\nexec coresmith \"$@\"\nEOF\nmv /work/cs.sh /work/scripts/cs", None)])
    assert {w["path"] for w in wrappers} == {"/work/cs.sh", "/work/scripts/cs"}
    calls = [
        {"id": "a:1", "start_ts": 10.0, "end_ts": 11.0, **cli_calls.parse_invocations("coresmith stage status --json")},
        {"id": "a:2", "start_ts": 20.0, "end_ts": 23.0, **cli_calls.parse_invocations("for k in prd sad; do coresmith schema $k; done")},
        {"id": "a:3", "start_ts": 21.0, "end_ts": 24.0, **cli_calls.parse_invocations("for k in frd ers; do coresmith schema $k; done")},
        {"id": "a:4", "start_ts": 30.0, "end_ts": 31.0, **cli_calls.parse_invocations("scripts/cs build status --json", wrappers)},
        {"id": "a:5", "start_ts": 40.0, "end_ts": 41.0, "exit_code": 0, **cli_calls.parse_invocations("coresmith build --help | head")},
        {"id": "a:6", "start_ts": 50.0, "end_ts": 51.0, **cli_calls.parse_invocations("echo coresmith build module cpu; which coresmith")},
    ]
    actions = [{"id": 1, "ts": 10.9, "argv": ["stage", "status", "--json"]},
               {"id": 2, "ts": 21.5, "argv": ["schema", "sad"]},
               {"id": 3, "ts": 22.0, "argv": ["schema", "frd"]},
               {"id": 4, "ts": 30.8, "argv": ["build", "status", "--json"]}]
    stats = cli_calls.match(actions, calls)
    assert [a["link"] for a in actions] == ["strong", "strong", "strong", "strong"]
    assert actions[1]["native"][0]["call"] == "a:2" and actions[2]["native"][0]["call"] == "a:3"
    assert actions[3]["native"][0]["call"] == "a:4"
    assert calls[3]["invocations"][0]["via_wrapper"] == "/work/scripts/cs"
    assert [m["code"] for m in calls[4]["missing"]] == ["help"]
    assert calls[5]["invocations"] == []
    assert {m["argv"][1] for m in calls[1]["missing"]} == {"prd"}, "the loop value with no audit row is reported"
    assert stats["strong"] == 4


def test_cli_through_a_variable_assigned_in_the_same_command():
    inv = cli_calls.parse_invocations('CS="${CORESMITH_CLI:-coresmith}"; "$CS" verify rtl sdram_ctrl --json > r.log 2>&1')
    assert [i["argv"] for i in inv["invocations"]] == [["verify", "rtl", "sdram_ctrl", "--json"]]
    assert inv["invocations"][0]["via_wrapper"].startswith("$CS")
    assert cli_calls.parse_invocations("X=ls; $X build status")["invocations"] == []


def test_audit_rows_are_not_linked_to_unrelated_background_scripts():
    calls = [{"id": "bg:1", "start_ts": 0.0, "end_ts": 600.0, **cli_calls.parse_invocations("tail -f uart.log | grep -E boot; python3 poll.py")}]
    actions = [{"id": 1, "ts": 100.0, "argv": ["build", "status"]}]
    cli_calls.match(actions, calls)
    assert actions[0]["link"] == "unlinked" and actions[0]["native"] == []
    assert actions[0]["context"][0]["call"] == "bg:1"


# ------------------------------------------------------------------ daemon epochs and helper identity
def test_call_index_restart_is_not_joined_across_daemon_processes(tmp_path):
    evs = [{"id": f"E{i}", "ts": ts, "pid": pid, "event": ev, "node": node, "fields": f, "block": None, "build_id": None}
           for i, (ts, pid, ev, node, f) in enumerate([
               (100.0, 15, "graph_node_enter", "Init Block", {}), (101.0, 15, "graph_node_exit", "Init Block", {}),
               (102.0, 15, "llm_start", "LLM", {"run_name": "x"}),
               (103.0, 501, "llm_call_start", "LLM", {"call_index": 1}),
               (500.0, 77, "graph_node_enter", "Init Block", {}), (501.0, 77, "graph_node_exit", "Init Block", {}),
               (502.0, 77, "llm_start", "LLM", {"run_name": "y"}),
               (503.0, 902, "llm_call_start", "LLM", {"call_index": 1})])]
    epochs = engine.daemon_epochs(evs, tmp_path)
    assert [(e["epoch"], e["pid"]) for e in epochs] == [(1, 15), (2, 77)]
    assert engine.epoch_for(epochs, 200.0)["pid"] == 15 and engine.epoch_for(epochs, 600.0)["pid"] == 77


# ------------------------------------------------------------------ namespaces and lineage
def test_namespace_ancestor_uses_segment_boundaries():
    s = lineage.ns_scope("process_block:9539|block_done:26ad", {"", "process_block:9539"})
    assert s["exact_member"] is False and s["checkpointed_ancestor"] == "process_block:9539"
    s = lineage.ns_scope("process_block:95|block_done:1", {"process_block:9539"})
    assert s["checkpointed_ancestor"] is None, "a longer id sharing a prefix is not an ancestor"
    s = lineage.ns_scope("a:1", {"a:1"})
    assert s["exact_member"] is True


def _build_row(bid, module, *, graph="build", thread=None, ns=None, status="completed", result=None, t=100.0):
    return (bid, module, "", "build_module" if graph == "build" else "run_start", graph, thread or f"build-{bid}", ns, status,
            t, t, t + 50, json.dumps({"digest": "d"}), "{}", None, json.dumps(result) if result else None, None)


def test_lineage_engine_reading_vs_all_files_reading(tmp_path):
    bid = "b-uart-20261008T105248-554bdf"
    res = {"attempt": 1, "thread_id": f"build-{bid}", "checkpoint_ns": "block_done:x", "timing_required": False}
    db = tmp_path / "project.sqlite"
    _project_db(db, builds=[_build_row(bid, "uart", result=res)],
                actions=[(1, 100.2, "architect", json.dumps(["build", "module", "uart", "--json"]), 0, "ok", ""),
                         (2, 160.0, "architect", json.dumps(["build", "status", "--build-id", bid]), 0, "ok", "")],
                dv=[(1, 140.0, "uart", "rtl", "gate", 1, 1, 0, 8, 8, bid, "")],
                cov=[(1, 141.0, "uart", "rtl", 10, 10, 100.0, json.dumps({"floor": 90, "passed": True}), bid, 1, "")],
                ppa=[(1, 142.0, "uart", 1, "gate", "synth", 120, None, 1, bid, "estimated", "")],
                results=[("uart", "best", json.dumps({"build_id": bid, "done": True}), None, 150.0)])
    led = Ledger()
    proj = engine.load_project(db, Sources(tmp_path), "t", led)
    rot = tmp_path / "pipeline_events.20261008-124808.jsonl"
    rot.write_text("\n".join(json.dumps(e) for e in [
        {"ts": 100.5, "pid": 14, "event": "graph_node_enter", "node": "Init Block", "block": "uart", "build_id": bid},
        {"ts": 100.6, "pid": 14, "event": "graph_node_exit", "node": "Init Block", "block": "uart", "build_id": bid},
        {"ts": 110.0, "pid": 14, "event": "graph_node_enter", "node": "Generate RTL", "block": "uart", "attempt": 1},
        {"ts": 111.0, "pid": 14, "event": "graph_node_exit", "node": "Generate RTL", "block": "uart", "attempt": 1},
        {"ts": 149.0, "pid": 14, "event": "graph_node_exit", "node": "Block Done", "block": "uart", "build_id": bid}]) + "\n")
    (tmp_path / "pipeline_events.jsonl").write_text(json.dumps({"ts": 300.0, "pid": 14, "event": "stage_done", "node": "Stage"}) + "\n")
    evs = engine.load_events(tmp_path, Sources(tmp_path), "t", led)["events"]
    ck = {"build": {"threads": {f"build-{bid}": {"": [{"id": "c1", "ts": 100.5, "ns": "", "thread": f"build-{bid}"}]}}}}
    out = lineage.build_lineage("t", proj, ck, evs, [], [], [])
    b = out["builds"][0]
    assert b["lineage"]["engine"]["ok"] is False
    assert any("init and completion" in r for r in b["lineage"]["engine"]["reasons"])
    assert b["lineage"]["viewer"]["ok"] is True, b["lineage"]["viewer"]["reasons"]
    assert b["events_in_rotated_files"] == 3 and len(b["events_inferred"]) == 2
    assert [s["node"] for s in b["segments"]] == ["Init Block", "Generate RTL", "Block Done"]
    labels = {x["action"]: x["label"] for x in b["actions"]}
    assert labels == {1: "strong", 2: "exact"}
    assert b["published"] is True


# ------------------------------------------------------------------ end to end
def _msgpack(o) -> bytes:
    if isinstance(o, dict):
        return bytes([0x80 | len(o)]) + b"".join(_msgpack(k) + _msgpack(v) for k, v in o.items())
    if isinstance(o, str):
        b = o.encode()
        return (bytes([0xA0 | len(b)]) if len(b) < 32 else bytes([0xD9, len(b)])) + b
    if isinstance(o, int):
        return bytes([o])
    raise TypeError(o)


def _snapshot(tmp_path: Path, extra_tool_output: str = "") -> Path:
    snap = tmp_path / "snap"
    arm = snap / "arms" / "demo"
    cs = arm / "work" / ".coresmith"
    cs.mkdir(parents=True)
    (arm / "config.json").write_text(json.dumps({"provider": "claude", "model": "claude-x", "effort": "high"}))
    (arm / "status.json").write_text(json.dumps({"session_id": "sess-1", "state": "running",
                                                 "run_started_at": "2026-10-08T07:00:00+00:00",
                                                 "last_event_at": "2026-10-08T08:00:00+00:00"}))
    inv = arm / "invocations" / "0001"
    inv.mkdir(parents=True)
    (inv / "command.json").write_text(json.dumps(["claude", "--env", "ANTHROPIC_API_KEY=supersecret"]))
    (inv / "status.json").write_text(json.dumps({"session_id": "sess-1"}))
    t0 = 1791442800.0                      # 2026-10-08T07:00:00Z
    recs = [
        {"type": "user", "uuid": "u0", "sessionId": "sess-1", "timestamp": "2026-10-08T07:00:01Z",
         "message": {"role": "user", "content": "Build the <img src=x onerror=alert(1)> chip"}},
        {"type": "assistant", "uuid": "u1", "sessionId": "sess-1", "timestamp": "2026-10-08T07:00:02Z",
         "message": {"id": "m1", "content": [{"type": "thinking", "thinking": THINK, "signature": SIG},
                                             {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                                              "input": {"command": "coresmith stage status --json"}}]}},
        {"type": "user", "uuid": "u2", "sessionId": "sess-1", "timestamp": "2026-10-08T07:00:04Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                                  "content": "</script><script>alert(2)</script>" + extra_tool_output}]}},
    ]
    sess = arm / "native-sessions" / ".claude" / "projects" / "-work"
    sess.mkdir(parents=True)
    (sess / "sess-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (inv / "transcript.jsonl").write_text("".join(json.dumps({**r, "session_id": "sess-1"}) + "\n" for r in recs[1:]) +
                                          json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 1.5,
                                                      "num_turns": 2, "session_id": "sess-1"}) + "\n")
    bid = "b-uart-20261008T070010-abcdef"
    _project_db(cs / "project.sqlite",
                actions=[(1, t0 + 3.5, "architect", json.dumps(["stage", "status", "--json"]), 0, "ok", "")],
                builds=[_build_row(bid, "uart", result={"attempt": 1, "thread_id": f"build-{bid}"}, t=t0 + 10)])
    con = sqlite3.connect(cs / "build_checkpoint.db")
    con.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, parent_checkpoint_id TEXT,"
                " type TEXT, checkpoint BLOB, metadata BLOB)")
    con.execute("CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, task_id TEXT, idx INTEGER,"
                " channel TEXT, type TEXT, value BLOB)")
    blob = _msgpack({"v": 4, "ts": "2026-10-08T07:00:11.000000+00:00", "versions_seen": {"init_block": {"x": 1}}})
    con.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)", (f"build-{bid}", "", "c1", None, "msgpack", blob,
                                                                  json.dumps({"source": "loop", "step": 1})))
    con.commit()
    con.close()
    (cs / "pipeline_events.jsonl").write_text(json.dumps({"ts": t0 + 10.5, "pid": 14, "event": "graph_node_enter",
                                                          "node": "Init Block", "block": "uart", "build_id": bid}) + "\n")
    return snap


def test_end_to_end_export_is_private_and_complete(tmp_path):
    snap = _snapshot(tmp_path)
    out = tmp_path / "viewer"
    summary = export.export(snap, out)
    assert summary["privacy_ok"] is True
    blob = "".join(p.read_text() for p in (out / "data").rglob("*.js"))
    assert THINK not in blob and SIG[:40] not in blob and "supersecret" not in blob
    assert "<img src=x onerror=alert(1)>" in blob, "record text is data: exported verbatim, rendered as text by the UI"
    idx = json.loads((out / "data" / "index.js").read_text().split(",", 1)[1].rsplit(")", 1)[0])
    a = idx["arms"]["demo"]
    assert a["counts"]["actions"] == 1 and a["actions"][0]["link"] == "strong"
    arch = next(x for x in a["agents"] if x["kind"] == "architect")
    assert arch["stats"]["duplicates_referenced"] >= 2, "stream copies fold into native-session turns"
    assert a["builds"][0]["checkpoints"]["count"] == 1
    for f in ("index.html", "app.js", "app.css"):
        assert (out / f).is_file()
    assert not list(snap.rglob("*-wal")) and not list(snap.rglob("*-shm")), "the snapshot is never modified"


def test_end_to_end_leak_in_unparsed_shape_fails_the_export(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path)
    # simulate a sanitiser regression: redaction of tool output disabled
    monkeypatch.setattr(native, "redact", lambda text, ledger=None: text)
    monkeypatch.setattr(native, "_claude_blocks_text", lambda content, ledger=None: json.dumps(
        [{"type": "text", "text": "dump"}, {"type": "x", "v": THINK}]))
    summary = export.export(snap, tmp_path / "viewer")
    assert summary["privacy_ok"] is False
    assert export.main(["--snapshot", str(snap), "--out", str(tmp_path / "viewer2")]) == 3


def test_verify_recounts_the_snapshot_and_detects_a_mismatch(tmp_path):
    from tools.run_viewer import verify
    snap = _snapshot(tmp_path)
    out = tmp_path / "viewer"
    export.export(snap, out)
    assert all(ok for _, ok, _ in verify.verify(snap, out))
    con = sqlite3.connect(snap / "arms" / "demo" / "work" / ".coresmith" / "project.sqlite")
    con.execute("INSERT INTO actions VALUES (99, 1.0, 'cli', '[\"status\"]', 0, '', '')")
    con.commit()
    con.close()
    assert not all(ok for _, ok, _ in verify.verify(snap, out))
