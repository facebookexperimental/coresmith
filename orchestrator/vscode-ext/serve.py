#!/usr/bin/env python3
"""
Standalone webview server for SoCMate.

Serves the same ReactFlow webview that the VS Code extension provides,
but as a regular web page in any browser. Calls the same graph
introspection code used by the MCP server.

Usage:
    python orchestrator/vscode-ext/serve.py [--port 3000]
"""

import argparse
import json
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

# Resolve project root: honour SOCMATE_PROJECT_ROOT env var (same as MCP server),
# then try .cursor/mcp.json, then fall back to relative path from this file.
import os as _os

def _resolve_project_root() -> Path:
    # 1. Env var (matches MCP server)
    env_root = _os.environ.get("SOCMATE_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    # 2. Read from .cursor/mcp.json in this workspace
    workspace = Path(__file__).resolve().parent.parent.parent
    mcp_cfg = workspace / ".cursor" / "mcp.json"
    if mcp_cfg.exists():
        try:
            import json as _json
            cfg = _json.loads(mcp_cfg.read_text())
            for srv in cfg.get("mcpServers", {}).values():
                root = (srv.get("env") or {}).get("SOCMATE_PROJECT_ROOT")
                if root:
                    return Path(root)
        except Exception:
            pass
    # 3. Fallback
    return workspace

PROJECT_ROOT = _resolve_project_root()
# Add both the data root and the code root to sys.path
_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if str(_CODE_ROOT) != str(PROJECT_ROOT):
    sys.path.insert(0, str(_CODE_ROOT))

# Initialise lightweight OTel tracing (no-op if already done)
from orchestrator.telemetry import init_telemetry  # noqa: E402

init_telemetry(str(PROJECT_ROOT))


def get_graph_data(graph_name: str) -> dict:
    """Introspect a LangGraph graph and return node/edge JSON."""
    from orchestrator.mcp_server import _introspect_graph
    # Use _CODE_ROOT for prompt file resolution (code + prompts live here,
    # which may differ from PROJECT_ROOT when data root is separate).
    return _introspect_graph(graph_name, str(_CODE_ROOT))


def get_execution_status(graph_filter: str = "") -> dict:
    """Read pipeline event log and derive per-node execution status.

    Args:
        graph_filter: If set, only return events matching this graph name.
            Events without a "graph" field are treated as "frontend".
    """
    events_file = PROJECT_ROOT / ".socmate" / "pipeline_events.jsonl"
    status = {}
    if not events_file.exists():
        return status
    try:
        for line in events_file.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            event_graph = e.get("graph", "frontend")
            if graph_filter and event_graph != graph_filter:
                continue
            node = e.get("node") or e.get("block")
            event_type = e.get("event", "")
            if node:
                if "start" in event_type or "enter" in event_type:
                    status[node] = "running"
                elif "end" in event_type or "exit" in event_type:
                    status[node] = "failed" if e.get("error") else "done"
    except Exception:
        pass
    return status


def get_live_calls(node_name: str) -> list[dict]:
    """Get LLM calls for a currently-running node by correlating timestamps.

    When a node span hasn't ended yet (so OTel traces aren't available),
    we correlate pipeline_events.jsonl (which records node enter/exit times)
    with llm_calls.jsonl (which is written after each LLM call completes)
    to find LLM calls that happened within the running node's time window.

    Returns a list in the same format as get_node_traces():
        [{"attempt": 1, "spans": [...]}]
    """
    events_file = PROJECT_ROOT / ".socmate" / "pipeline_events.jsonl"
    llm_log = PROJECT_ROOT / ".socmate" / "llm_calls.jsonl"

    if not events_file.exists():
        return []

    # Find ALL enter/exit windows for this node (across all blocks and
    # invocations).  Architecture nodes like Gather Requirements run
    # multiple times; using a unique key per invocation prevents later
    # windows from overwriting earlier ones.
    # Each window also stores the attempt number from the event so that
    # results can be grouped by (block, attempt) for the detail panel.
    block_windows: dict[str, dict] = {}
    open_stack: dict[str, str] = {}  # block -> current open window key
    window_counter = 0
    for line in events_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_node = e.get("node", "")
        event_type = e.get("event", "")
        # Use the same block resolution as get_timeline_data:
        # pipeline events have "block", architecture events use "graph".
        block = e.get("block") or e.get("graph", "")
        ts = e.get("ts")
        if event_node != node_name or ts is None:
            continue
        block_key = block or "__default__"
        if "enter" in event_type:
            # If a previous enter for this (node, block) never got its exit
            # event, treat it as stale rather than leaving it open: an
            # always-open window otherwise matches *every* later LLM call in
            # llm_calls.jsonl, badly over-attributing calls to this node.
            prev_key = open_stack.get(block_key)
            if (prev_key and prev_key in block_windows
                    and block_windows[prev_key].get("exit_ts") is None):
                block_windows.pop(prev_key, None)
            window_counter += 1
            key = f"{block_key}_{window_counter}"
            attempt = (e.get("attempt") or e.get("round")
                       or window_counter)
            block_windows[key] = {
                "enter_ts": ts, "exit_ts": None,
                "block": block, "attempt": attempt,
            }
            open_stack[block_key] = key
        elif "exit" in event_type and block_key in open_stack:
            key = open_stack.pop(block_key)
            if key in block_windows:
                block_windows[key]["exit_ts"] = ts

    # Only keep windows that are still open (no exit) or recently closed
    open_windows = {k: w for k, w in block_windows.items()
                    if w.get("exit_ts") is None}

    if not open_windows and not block_windows:
        return []

    # Use open windows if available, otherwise fall back to all windows
    windows = open_windows if open_windows else block_windows

    if not llm_log.exists():
        return []

    # Read LLM calls and match by timestamp
    all_calls = []
    for line in llm_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        call_ts = record.get("ts")
        if call_ts is None:
            continue

        # Check if this call falls within any of our windows
        for key, window in windows.items():
            enter_ts = window["enter_ts"]
            exit_ts = window.get("exit_ts")
            if call_ts >= enter_ts and (exit_ts is None or call_ts <= exit_ts):
                duration_ms = record.get("duration_s", 0) * 1000
                all_calls.append({
                    "span_id": f"live_{call_ts}",
                    "name": f"LLM {record.get('model', 'unknown')}",
                    "status": "error" if record.get("error") else "ok",
                    "duration_ms": round(duration_ms, 2),
                    "attributes": {
                        "llm.model_name": record.get("model", "unknown"),
                        "llm.provider": record.get("provider", ""),
                        "input.value": (
                            record.get("system_prompt", "")
                            + "\n---\n"
                            + record.get("user_prompt", "")
                        )[:32000],
                        "output.value": record.get("response", "")[:32000],
                        "block_name": window.get("block", ""),
                    },
                    "children": [],
                    "_window_key": key,
                })
                break

    # Check for active or recently-done streaming LLM calls
    live_dir = PROJECT_ROOT / ".socmate" / "live_streams"
    _DONE_FILE_MAX_AGE_S = 120  # clean up done files after 2 minutes
    _ORPHAN_MAX_AGE_S = 300     # clean up non-done files whose PID is dead
    orphan_candidates = []      # files to check for orphan cleanup after matching
    if live_dir.is_dir():
        now = time.time()
        for stream_file in live_dir.glob("*.json"):
            try:
                stream_data = json.loads(stream_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            # Clean up stale done files
            is_done = stream_data.get("done", False)
            if is_done:
                done_ts = stream_data.get("done_ts", 0)
                if now - done_ts > _DONE_FILE_MAX_AGE_S:
                    try:
                        stream_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
            stream_ts = stream_data.get("started_ts")
            if stream_ts is None:
                continue
            matched = False
            for key, window in windows.items():
                enter_ts = window["enter_ts"]
                exit_ts = window.get("exit_ts")
                if stream_ts >= enter_ts and (exit_ts is None or stream_ts <= exit_ts):
                    matched = True
                    call_status = "ok" if is_done else "streaming"
                    all_calls.append({
                        "span_id": f"stream_{stream_data.get('pid', 0)}",
                        "name": f"LLM {stream_data.get('model', 'unknown')}",
                        "status": call_status,
                        "duration_ms": round(stream_data.get("elapsed_s", 0) * 1000, 2),
                        "attributes": {
                            "llm.model_name": stream_data.get("model", "unknown"),
                            "streaming": not is_done,
                            "block_name": window.get("block", ""),
                            "output.value": stream_data.get("partial_stdout", ""),
                        },
                        "children": [],
                        "_window_key": key,
                    })
                    break
            # Track unmatched non-done files for orphan cleanup
            if not matched and not is_done:
                orphan_candidates.append((stream_file, stream_data))

    # Clean up orphan stream files: non-done files whose PID is no
    # longer running and that didn't match any event window.
    for stream_file, stream_data in orphan_candidates:
        started = stream_data.get("started_ts", 0)
        if now - started > _ORPHAN_MAX_AGE_S:
            pid = stream_data.get("pid")
            if pid:
                try:
                    _os.kill(pid, 0)
                except (ProcessLookupError, OSError):
                    try:
                        stream_file.unlink(missing_ok=True)
                    except OSError:
                        pass

    if not all_calls:
        return []

    # Group by window key (block + invocation) so each attempt of the
    # same block gets its own tab in the detail panel.  Previously all
    # calls for a block were merged into one entry, hiding per-attempt
    # LLM data.
    by_window: dict[str, list] = {}
    for call in all_calls:
        wk = call.pop("_window_key", "")
        by_window.setdefault(wk, []).append(call)

    result = []
    # Ensure unique attempt numbers per (block, attempt) to avoid
    # duplicate tab keys in the webview.  Architecture nodes can have
    # the same round number across multiple invocations.
    seen_attempts: dict[str, int] = {}  # "block:attempt" -> max suffix
    for window_key, calls in sorted(by_window.items()):
        window = windows.get(window_key, {})
        block_name = window.get("block", "")
        attempt = window.get("attempt", 1)
        attempt_key = f"{block_name}:{attempt}"
        if attempt_key in seen_attempts:
            seen_attempts[attempt_key] += 1
            attempt = attempt + seen_attempts[attempt_key]
        else:
            seen_attempts[attempt_key] = 0
        has_streaming = any(c.get("status") == "streaming" for c in calls)
        result.append({
            "attempt": attempt,
            "spans": [{
                "span_id": f"live_root_{window_key}",
                "name": f"{node_name} [{block_name}]" if block_name else node_name,
                "node_name": node_name,
                "status": "ok",
                "duration_ms": None,
                "attributes": {"block_name": block_name, "live": True,
                               "attempt": attempt},
                "children": calls,
            }],
            "live": True,
            "has_streaming": has_streaming,
        })

    return result


def get_timeline_data(graph_filter: str = "") -> dict:
    """Parse events into Gantt timeline segments for any graph.

    Supports all three graphs: architecture, frontend (pipeline), backend.
    Architecture events use "graph" field as the block name (single row).
    Frontend/backend events use "block" field (one row per RTL block).

    Args:
        graph_filter: If set, only include events from this graph.
            Events without a "graph" field are treated as "frontend".

    Returns a dict with:
      - blocks: list of block dicts, each with name, tier, graph, status,
        and attempts containing node-level execution segments.
      - pipeline_start / pipeline_end: epoch timestamps bounding the run.
      - graph: the graph filter applied (or "all").
    """
    events_file = PROJECT_ROOT / ".socmate" / "pipeline_events.jsonl"
    empty = {"blocks": [], "pipeline_start": None, "pipeline_end": None,
             "graph": graph_filter or "all"}
    if not events_file.exists():
        return empty

    events = []
    for line in events_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not events:
        return empty

    # Filter by graph if requested
    if graph_filter:
        events = [e for e in events
                  if e.get("graph", "frontend") == graph_filter]

    if not events:
        return empty

    pipeline_start = events[0]["ts"]
    pipeline_end = events[-1]["ts"]

    blocks: dict[str, dict] = {}
    current_attempt: dict[str, int] = {}
    open_segments: dict[tuple[str, str], dict] = {}

    # Architecture completion nodes
    ARCH_COMPLETE_NODES = {"Architecture Complete", "Abort"}
    # Pipeline/backend completion nodes
    PIPELINE_COMPLETE_NODES = {"Advance Block", "Pipeline Complete",
                               "Backend Complete"}

    for e in events:
        event_type = e.get("event", "")
        event_graph = e.get("graph", "frontend")
        node = e.get("node")
        ts = e.get("ts")

        # Determine the block (row) key:
        # - Architecture events: use graph name as block (single row)
        # - Frontend/backend: use the RTL block name
        block = e.get("block") or event_graph
        if not block or not ts:
            continue

        if block not in blocks:
            # Friendly display name for architecture row
            display_name = block
            if block == "architecture":
                display_name = "Architecture"
            elif block == "backend":
                display_name = "Backend"

            blocks[block] = {
                "name": display_name,
                "tier": e.get("tier"),
                "graph": event_graph,
                "start_ts": ts,
                "end_ts": ts,
                "status": "running",
                "attempts": [],
            }
            current_attempt[block] = e.get("round", 1)

        blocks[block]["end_ts"] = ts

        if event_type == "block_start":
            attempt = e.get("attempt", 1)
            current_attempt[block] = attempt
            if e.get("tier"):
                blocks[block]["tier"] = e["tier"]

        # For architecture, use "round" as the attempt number
        if event_type == "graph_node_enter" and node:
            attempt = (e.get("attempt")
                       or e.get("round")
                       or current_attempt.get(block, 1))
            # Skip duplicate enters (e.g. LangGraph re-executes nodes
            # on interrupt resume, emitting a second enter event --
            # keep the original start time so the segment isn't lost).
            if (block, node) not in open_segments:
                open_segments[(block, node)] = {
                    "node": node,
                    "start_ts": ts,
                    "attempt": attempt,
                }

        elif event_type == "graph_node_exit" and node:
            key = (block, node)
            if key not in open_segments:
                continue
            seg = open_segments.pop(key)
            status = "done"
            result: dict = {}

            for flag in ("clean", "passed", "success", "all_pass"):
                val = e.get(flag)
                if val is not None:
                    result[flag] = val
                    if val is False:
                        status = "failed"

            if e.get("category"):
                result["category"] = e["category"]
            if e.get("chars"):
                result["chars"] = e["chars"]
            if e.get("gate_count") is not None:
                result["gate_count"] = e["gate_count"]
            if e.get("elapsed_s") is not None:
                result["elapsed_s"] = e["elapsed_s"]
            if e.get("violations") is not None:
                result["violations"] = e["violations"]
            if e.get("blocks") is not None:
                result["block_count"] = e["blocks"]

            # Capture additional metadata from exit events so the
            # detail panel can show useful info for non-LLM nodes
            # (e.g. Create Documentation, Finalize Architecture).
            _META_KEYS = ("node_count", "edge_count",
                          "validation_errors", "path", "error",
                          "round", "peripheral_count", "questions")
            metadata = {}
            for mk in _META_KEYS:
                mv = e.get(mk)
                if mv is not None:
                    metadata[mk] = mv
            if metadata:
                result["metadata"] = metadata

            segment = {
                "node": seg["node"],
                "start_ts": seg["start_ts"],
                "end_ts": ts,
                "duration_s": round(ts - seg["start_ts"], 3),
                "status": status,
                "attempt": seg["attempt"],
                **result,
            }

            attempt_num = seg["attempt"]
            block_data = blocks[block]
            attempt_entry = next(
                (a for a in block_data["attempts"] if a["attempt"] == attempt_num),
                None,
            )
            if not attempt_entry:
                attempt_entry = {"attempt": attempt_num, "segments": []}
                block_data["attempts"].append(attempt_entry)
            attempt_entry["segments"].append(segment)

            # Mark completion
            if node in PIPELINE_COMPLETE_NODES:
                block_data["status"] = "done" if e.get("success") else "failed"
            elif node in ARCH_COMPLETE_NODES:
                block_data["status"] = "done" if node == "Architecture Complete" else "failed"

    # Nodes that pause for human-in-the-loop review
    HITL_NODES = {"Review Uarch Spec", "Ask Human",
                  "Escalate PRD", "Escalate Diagram",
                  "Escalate Constraints", "Escalate Exhausted"}

    # Add still-open (running) segments
    for (block, node), seg in open_segments.items():
        if block not in blocks:
            continue
        attempt_num = seg["attempt"]
        block_data = blocks[block]
        attempt_entry = next(
            (a for a in block_data["attempts"] if a["attempt"] == attempt_num),
            None,
        )
        if not attempt_entry:
            attempt_entry = {"attempt": attempt_num, "segments": []}
            block_data["attempts"].append(attempt_entry)
        seg_status = "waiting" if node in HITL_NODES else "running"
        attempt_entry["segments"].append({
            "node": seg["node"],
            "start_ts": seg["start_ts"],
            "end_ts": None,
            "duration_s": None,
            "status": seg_status,
            "attempt": attempt_num,
        })

    # Count blocks waiting for human review
    waiting_for_human = 0
    active_blocks = 0
    for (blk, nd) in open_segments:
        if blk in blocks:
            if nd in HITL_NODES:
                waiting_for_human += 1
            else:
                active_blocks += 1

    block_list = sorted(blocks.values(), key=lambda b: b["start_ts"])
    return {
        "blocks": block_list,
        "pipeline_start": pipeline_start,
        "pipeline_end": pipeline_end,
        "graph": graph_filter or "all",
        "waiting_for_human": waiting_for_human,
        "active_blocks": active_blocks,
    }


def get_block_diagram_viz() -> dict:
    """Read the block diagram visualization JSON produced by the Create Documentation agent."""
    viz_path = PROJECT_ROOT / ".socmate" / "block_diagram_viz.json"
    if not viz_path.exists():
        return {}
    try:
        return json.loads(viz_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_summary(stage: str) -> dict:
    """Read an observer-generated summary markdown file for the given stage."""
    valid_stages = ("architecture", "frontend", "backend")
    if stage not in valid_stages:
        return {"stage": stage, "content": "", "updated": None,
                "error": f"Invalid stage. Must be one of: {', '.join(valid_stages)}"}
    summary_path = PROJECT_ROOT / "arch" / f"summary_{stage}.md"
    if summary_path.exists():
        try:
            content = summary_path.read_text(encoding="utf-8")
            updated = summary_path.stat().st_mtime
            return {"stage": stage, "content": content, "updated": updated}
        except OSError:
            return {"stage": stage, "content": "", "updated": None}
    return {"stage": stage, "content": "", "updated": None}


def _read_uarch_specs_for_cards() -> dict:
    """Read uArch spec files and return structured data for frontend cards."""
    import re as _re

    specs_dir = PROJECT_ROOT / "arch" / "uarch_specs"
    if not specs_dir.is_dir():
        return {}

    results = {}
    for spec_file in sorted(specs_dir.glob("*.md")):
        block_name = spec_file.stem
        try:
            content = spec_file.read_text(encoding="utf-8")
        except OSError:
            continue

        summary = {}
        json_match = _re.search(r"```json\s*\n(.*?)```", content, _re.DOTALL)
        if json_match:
            try:
                summary = json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        overview = ""
        overview_match = _re.search(
            r"##\s*1\.\s*Block\s+Overview\s*\n(.*?)(?=\n##\s|\Z)",
            content, _re.DOTALL,
        )
        if overview_match:
            raw = overview_match.group(1).strip()
            first_para = raw.split("\n\n")[0].strip()
            overview = first_para[:500]

        results[block_name] = {"summary": summary, "overview": overview, "full_content": content}

    return results


def get_summary_cards(stage: str) -> dict:
    """Return structured card data for the summary panel.

    Architecture stage returns: summary, prd (from prd_spec.json), ers_content.
    Frontend stage returns: summary, uarch_specs.
    Backend stage returns: summary (plain markdown).
    """
    summary_path = PROJECT_ROOT / "arch" / f"summary_{stage}.md"
    summary_text = ""
    summary_updated = None
    if summary_path.exists():
        try:
            summary_text = summary_path.read_text(encoding="utf-8")
            summary_updated = summary_path.stat().st_mtime
        except OSError:
            pass

    if stage == "architecture":
        def _read_md(name: str) -> str:
            p = PROJECT_ROOT / "arch" / name
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8")
                except OSError:
                    pass
            return ""

        # PRD .md is the canonical human-readable source; ERS .md is
        # a separate doc produced later by create_documentation_node.
        prd_content = _read_md("prd_spec.md")
        sad_content = _read_md("sad_spec.md")
        frd_content = _read_md("frd_spec.md")
        ers_content = _read_md("ers_spec.md")
        clock_tree_content = _read_md("clock_tree.md")
        memory_map_content = _read_md("memory_map.md")

        return {
            "stage": "architecture",
            "updated": summary_updated,
            "summary": summary_text,
            "prd_content": prd_content,
            "sad_content": sad_content,
            "frd_content": frd_content,
            "ers_content": ers_content,
            "clock_tree_content": clock_tree_content,
            "memory_map_content": memory_map_content,
        }

    elif stage == "frontend":
        uarch_specs = _read_uarch_specs_for_cards()
        return {
            "stage": "frontend",
            "updated": summary_updated,
            "summary": summary_text,
            "uarch_specs": uarch_specs,
        }

    if stage == "backend":
        return _get_backend_cards(summary_text, summary_updated)

    return {
        "stage": stage,
        "updated": summary_updated,
        "summary": summary_text,
    }


def _get_backend_cards(summary_text: str, summary_updated) -> dict:
    """Build structured backend card data from backend_results.json and reports."""
    results_path = PROJECT_ROOT / ".socmate" / "backend_results.json"
    blocks: list[dict] = []
    target_clock_mhz = 0.0

    if results_path.exists():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            target_clock_mhz = data.get("target_clock_mhz", 0)
            blocks = data.get("blocks", [])
        except (json.JSONDecodeError, OSError):
            pass

    if not blocks:
        blocks = _build_backend_blocks_from_events()

    # Read gate counts from frontend synthesis events
    gate_counts = _read_gate_counts()

    # Compute max frequency from WNS + target clock for each block
    for blk in blocks:
        name = blk.get("name", "")
        if name in gate_counts:
            blk["gate_count"] = gate_counts[name]
        wns = blk.get("wns_ns", blk.get("timing_wns_ns", 0))
        if target_clock_mhz > 0 and wns != 0:
            period_ns = 1000.0 / target_clock_mhz
            actual_period = period_ns - wns
            if actual_period > 0:
                blk["max_freq_mhz"] = round(1000.0 / actual_period, 1)
        elif target_clock_mhz > 0:
            blk["max_freq_mhz"] = target_clock_mhz
        # Image paths: convert absolute to API paths
        for img_key in ("floorplan_image", "gds_image"):
            path = blk.get(img_key, "")
            if path:
                p = Path(path)
                if p.exists():
                    try:
                        blk[img_key] = str(p.relative_to(PROJECT_ROOT))
                    except ValueError:
                        pass

    return {
        "stage": "backend",
        "updated": summary_updated,
        "summary": summary_text,
        "target_clock_mhz": target_clock_mhz,
        "blocks": blocks,
    }


def _read_gate_counts() -> dict[str, int]:
    """Read gate counts from pipeline events (frontend synthesis results)."""
    events_file = PROJECT_ROOT / ".socmate" / "pipeline_events.jsonl"
    counts: dict[str, int] = {}
    if not events_file.exists():
        return counts
    try:
        for line in events_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("node") == "Synthesize" and ev.get("gate_count"):
                block = ev.get("block", "")
                if block:
                    counts[block] = ev["gate_count"]
    except OSError:
        pass
    return counts


def _build_backend_blocks_from_events() -> list[dict]:
    """Fallback: build block list from backend pipeline events."""
    events_file = PROJECT_ROOT / ".socmate" / "pipeline_events.jsonl"
    blocks: dict[str, dict] = {}
    if not events_file.exists():
        return []
    try:
        for line in events_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("graph") != "backend":
                continue
            block = ev.get("block", "")
            if not block:
                continue
            if block not in blocks:
                blocks[block] = {"name": block, "success": False}
            node = ev.get("node", "")
            if node == "Run PnR" and ev.get("type") == "graph_node_exit":
                blocks[block].update({
                    "success": ev.get("success", False),
                    "design_area_um2": ev.get("design_area_um2", 0),
                    "utilization_pct": ev.get("utilization_pct", 0),
                    "wns_ns": ev.get("wns_ns", 0),
                    "tns_ns": ev.get("tns_ns", 0),
                    "total_power_mw": ev.get("total_power_mw", 0),
                })
            if node == "Advance Block" and ev.get("type") == "graph_node_exit":
                blocks[block]["success"] = ev.get("success", False)
    except OSError:
        pass
    return list(blocks.values())


def get_uarch_spec(block_name: str) -> dict:
    """Read a uarch spec markdown file for the given block."""
    import re as _re
    if not _re.fullmatch(r"[a-zA-Z0-9_\-]+", block_name):
        return {"block_name": block_name, "content": "", "updated": None,
                "error": f"Invalid block name: '{block_name}'"}
    spec_dir = (PROJECT_ROOT / "arch" / "uarch_specs").resolve()
    spec_path = (spec_dir / f"{block_name}.md").resolve()
    if not spec_path.is_relative_to(spec_dir):
        return {"block_name": block_name, "content": "", "updated": None,
                "error": "Path traversal rejected"}
    if spec_path.exists():
        try:
            content = spec_path.read_text(encoding="utf-8")
            updated = spec_path.stat().st_mtime
            return {"block_name": block_name, "content": content, "updated": updated}
        except OSError:
            return {"block_name": block_name, "content": "", "updated": None}
    return {"block_name": block_name, "content": "", "updated": None,
            "error": f"No uarch spec found for '{block_name}'"}


def get_pending_interrupts() -> dict:
    """Identify blocks waiting at HITL nodes and architecture escalation history.

    Parses the event log to find:
    - Open segments at pipeline HITL nodes (Review Uarch Spec, Ask Human)
    - Architecture escalation events (Escalate PRD, Constraints, Diagram, Exhausted)
      with their questions, violations, and responses.
    """
    PIPELINE_HITL = {"Review Uarch Spec", "Ask Human"}
    ARCH_ESCALATION = {"Escalate PRD", "Escalate Diagram",
                       "Escalate Constraints", "Escalate Exhausted"}
    ARCH_PHASE_MAP = {
        "Escalate PRD": "prd",
        "Escalate Diagram": "block_diagram",
        "Escalate Constraints": "constraints",
        "Escalate Exhausted": "max_rounds_exhausted",
    }
    ARCH_ACTIONS_MAP = {
        "Escalate PRD": ["continue", "abort"],
        "Escalate Diagram": ["continue", "feedback", "abort"],
        "Escalate Constraints": ["retry", "accept", "feedback", "abort"],
        "Escalate Exhausted": ["retry", "accept", "feedback", "abort"],
    }

    events_file = PROJECT_ROOT / ".socmate" / "pipeline_events.jsonl"

    if not events_file.exists():
        return {"interrupts": [], "count": 0}

    # Track open segments for pipeline HITL nodes
    open_segments: dict[tuple[str, str], dict] = {}
    # Track architecture escalation events (enter/exit pairs)
    arch_entries: list[dict] = []
    arch_open: dict[str, int] = {}  # node -> index in arch_entries
    # Track escalation response events
    escalation_responses: list[dict] = []

    try:
        for line in events_file.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            node = e.get("node")
            event_type = e.get("event", "")
            block = e.get("block", "")
            ts = e.get("ts")
            if not node or not ts:
                continue

            if node in PIPELINE_HITL:
                key = (block, node)
                if "enter" in event_type:
                    open_segments[key] = {"block": block, "node": node, "ts": ts}
                elif "exit" in event_type:
                    open_segments.pop(key, None)

            elif node in ARCH_ESCALATION:
                if "enter" in event_type:
                    entry = {
                        "node": node,
                        "enter_ts": ts,
                        "enter_data": e,
                        "exit_data": None,
                        "status": "waiting",
                    }
                    arch_open[node] = len(arch_entries)
                    arch_entries.append(entry)
                elif "exit" in event_type and node in arch_open:
                    idx = arch_open.pop(node)
                    arch_entries[idx]["exit_data"] = e
                    arch_entries[idx]["exit_ts"] = ts
                    arch_entries[idx]["status"] = "completed"

            if event_type == "escalation_response":
                escalation_responses.append(e)
    except Exception:
        return {"interrupts": [], "count": 0}

    interrupts: list[dict] = []

    # --- Pipeline HITL entries (existing behaviour) ---
    for (block, node), seg in open_segments.items():
        if node not in PIPELINE_HITL or not block:
            continue

        entry = {
            "block_name": block,
            "node": node,
            "waiting_since": seg["ts"],
            "type": "uarch_spec_review" if node == "Review Uarch Spec" else "human_intervention",
            "supported_actions": (
                ["approve", "revise", "skip"]
                if node == "Review Uarch Spec"
                else ["retry", "fix_rtl", "add_constraint", "skip", "abort"]
            ),
        }

        spec_path = PROJECT_ROOT / "arch" / "uarch_specs" / f"{block}.md"
        if spec_path.exists():
            try:
                entry["spec_content"] = spec_path.read_text(encoding="utf-8")
                entry["spec_path"] = str(spec_path)
            except OSError:
                pass

        interrupts.append(entry)

    # --- Architecture escalation entries ---
    for esc in arch_entries:
        enter = esc["enter_data"]
        exit_d = esc.get("exit_data") or {}
        node = esc["node"]

        entry: dict = {
            "node": node,
            "type": "architecture_escalation",
            "phase": ARCH_PHASE_MAP.get(node, "unknown"),
            "status": esc["status"],
            "waiting_since": esc.get("enter_ts"),
            "round": enter.get("round"),
            "supported_actions": ARCH_ACTIONS_MAP.get(node, []),
            # Content from enter event (questions, violations, etc.)
            "questions": enter.get("questions", []),
            "violations": enter.get("violations", []),
            "structural_violations": enter.get("structural_violations", []),
            "question_count": enter.get("question_count", 0),
            "structural_count": enter.get("structural_count", 0),
            "total_violations": enter.get("total_violations", 0),
            "max_rounds": enter.get("max_rounds"),
        }

        if exit_d:
            resp: dict = {
                "action": exit_d.get("action"),
                "has_answers": exit_d.get("has_answers", False),
                "answer_count": exit_d.get("answer_count", 0),
                "answer_keys": exit_d.get("answer_keys", []),
            }
            # Include feedback text if present in the exit event
            fb = exit_d.get("feedback", "")
            if fb:
                resp["feedback"] = fb
            # Fallback: check escalation_response events for feedback text
            if not fb and escalation_responses:
                enter_ts = esc.get("enter_ts", 0)
                exit_ts = esc.get("exit_ts", float("inf"))
                for er in escalation_responses:
                    er_ts = er.get("ts", 0)
                    if enter_ts <= er_ts <= exit_ts and er.get("feedback"):
                        resp["feedback"] = er["feedback"]
                        break
            entry["response"] = resp

        # For PRD, try to attach answered Q&A from the PRD spec on disk
        if node == "Escalate PRD" and esc["status"] == "completed":
            prd_path = PROJECT_ROOT / ".socmate" / "prd_spec.json"
            if prd_path.exists():
                try:
                    prd_data = json.loads(prd_path.read_text(encoding="utf-8"))
                    entry["prd_answers"] = prd_data.get("user_answers", {})
                except (json.JSONDecodeError, OSError):
                    pass

        interrupts.append(entry)

    return {"interrupts": interrupts, "count": len(interrupts)}


DIST_DIR = Path(__file__).resolve().parent / "dist"

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SoCMate</title>
  <link rel="stylesheet" href="/dist/webview.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body, #root { width: 100%; height: 100%; overflow: hidden; }
    body {
      font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;
    }
  </style>
  <script>
    (function() {
      try {
        var t = localStorage.getItem('socmate-theme');
        if (t === 'dark' || t === 'light') {
          document.documentElement.setAttribute('data-theme', t);
        } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
          document.documentElement.setAttribute('data-theme', 'dark');
        }
      } catch(e) {}
    })();
  </script>
</head>
<body>
  <div id="root"></div>
  <script>
    window.onerror = function(msg, url, line, col, err) {
      var d = document.createElement('pre');
      d.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#fee;color:#c00;padding:16px;font-size:12px;border-bottom:2px solid #c00;white-space:pre-wrap;max-height:200px;overflow:auto';
      d.textContent = 'JS ERROR: ' + msg + '\\nat ' + url + ':' + line + ':' + col + '\\n' + (err && err.stack || '');
      document.body.prepend(d);
    };
  </script>
  <script type="module" src="/dist/webview.js"></script>
</body>
</html>"""


class WebviewHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the webview + API endpoints."""

    def do_HEAD(self):
        # Route HEAD through do_GET, suppressing the body. SimpleHTTPRequestHandler's
        # default do_HEAD would otherwise hit send_head() and try to serve from the
        # working directory, which bypasses our routing and leaks a directory listing.
        real_wfile = self.wfile

        class _NoBody:
            __slots__ = ("_w", "_suppress")
            def __init__(self, w):
                self._w = w
                self._suppress = False
            def write(self, data):
                if self._suppress:
                    try:
                        return len(data)
                    except TypeError:
                        return 0
                return self._w.write(data)
            def flush(self):
                return self._w.flush()

        wrapper = _NoBody(real_wfile)
        self.wfile = wrapper
        real_end_headers = self.end_headers

        def _end_headers():
            real_end_headers()
            wrapper._suppress = True

        self.end_headers = _end_headers
        try:
            self.do_GET()
        finally:
            self.wfile = real_wfile

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # API: graph introspection
        if path.startswith("/api/graph/"):
            graph_name = path.split("/api/graph/", 1)[1].strip("/") or "frontend"
            try:
                data = get_graph_data(graph_name)
                self._json_response(data)
            except KeyError:
                from orchestrator.mcp_server import _GRAPH_MODULES
                self._json_response(
                    {
                        "error": f"Unknown graph: {graph_name!r}",
                        "available": sorted(_GRAPH_MODULES.keys()),
                    },
                    status=404,
                )
            except Exception as exc:
                self._json_response({"error": str(exc)}, status=500)
            return

        # API: execution status (optional ?graph= filter)
        if path == "/api/status":
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            graph_filter = qs.get("graph", [""])[0]
            self._json_response(get_execution_status(graph_filter))
            return

        # API: timeline data for Gantt view (optional ?graph= filter)
        if path == "/api/timeline":
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            graph_filter = qs.get("graph", [""])[0]
            self._json_response(get_timeline_data(graph_filter))
            return

        # API: block diagram visualization JSON
        if path == "/api/block_diagram_viz":
            self._json_response(get_block_diagram_viz())
            return

        # API: uarch spec for a specific block
        if path.startswith("/api/uarch_spec/"):
            block_name = unquote(path.split("/api/uarch_spec/", 1)[1].strip("/"))
            if not block_name:
                self._json_response({"error": "block_name required"}, status=400)
                return
            self._json_response(get_uarch_spec(block_name))
            return

        # API: pending HITL interrupts with spec content
        if path == "/api/interrupts":
            self._json_response(get_pending_interrupts())
            return

        # API: observer summary for a pipeline stage
        if path.startswith("/api/summary/"):
            stage = path.split("/api/summary/", 1)[1].strip("/")
            self._json_response(get_summary(stage))
            return

        # API: structured card data for summary panel
        if path.startswith("/api/summary_cards/"):
            stage = path.split("/api/summary_cards/", 1)[1].strip("/")
            self._json_response(get_summary_cards(stage))
            return

        # API: live LLM calls for a running node (falls back when OTel traces
        # are not yet available because the parent span hasn't ended)
        if path.startswith("/api/live_calls/"):
            node_name = unquote(path.split("/api/live_calls/", 1)[1].strip("/"))
            if not node_name:
                self._json_response({"error": "node_name required"}, status=400)
                return
            try:
                self._json_response(get_live_calls(node_name))
            except Exception as exc:
                self._json_response({"error": str(exc)}, status=500)
            return

        # API: OTel traces for a graph node
        if path.startswith("/api/traces/"):
            node_name = unquote(path.split("/api/traces/", 1)[1].strip("/"))
            if not node_name:
                self._json_response({"error": "node_name required"}, status=400)
                return
            try:
                from orchestrator.telemetry.reader import get_node_traces

                db_path = str(PROJECT_ROOT / ".socmate" / "traces.db")
                traces = get_node_traces(db_path, node_name)
                self._json_response(traces)
            except Exception as exc:
                self._json_response({"error": str(exc)}, status=500)
            return

        # API: serve artifact files (images, reports) from project directory
        if path.startswith("/api/artifacts/"):
            rel = unquote(path.split("/api/artifacts/", 1)[1].strip("/"))
            if not rel or ".." in rel:
                self.send_error(403)
                return
            file_path = (PROJECT_ROOT / rel).resolve()
            if not file_path.is_relative_to(PROJECT_ROOT.resolve()):
                self.send_error(403)
                return
            if file_path.exists() and file_path.is_file():
                self._serve_file(file_path)
                return
            self.send_error(404)
            return

        # Serve built JS/CSS from dist/
        if path.startswith("/dist/"):
            rel = path[len("/dist/"):]
            file_path = (DIST_DIR / rel).resolve()
            if not file_path.is_relative_to(DIST_DIR.resolve()):
                self.send_error(403)
                return
            if file_path.exists() and file_path.is_file():
                self._serve_file(file_path)
                return
            self.send_error(404)
            return

        # Root: serve the index HTML
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode())
            return

        self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path):
        ext = file_path.suffix.lower()
        mime = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".json": "application/json",
            ".map": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "application/octet-stream")

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Quieter logging for polling endpoints
        msg = fmt % args
        if "/api/status" not in msg and "/api/timeline" not in msg and "/api/summary" not in msg and "/api/summary_cards" not in msg and "/api/block_diagram_viz" not in msg and "/api/interrupts" not in msg:
            print(f"  {msg}")


def main():
    parser = argparse.ArgumentParser(description="SoCMate — standalone server")
    parser.add_argument("--port", "-p", type=int, default=3000, help="Port (default: 3000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    # Verify dist exists
    if not (DIST_DIR / "webview.js").exists():
        print("ERROR: dist/webview.js not found. Run 'npm run build' first:")
        print(f"  cd {DIST_DIR.parent} && npm run build")
        sys.exit(1)

    httpd = HTTPServer((args.host, args.port), WebviewHandler)
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  SoCMate                                     ║")
    print(f"║  http://{args.host}:{args.port}                   ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"  Serving webview from {DIST_DIR}")
    print(f"  Graph API at /api/graph/frontend")
    print(f"  Press Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
