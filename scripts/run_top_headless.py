#!/usr/bin/env python3
"""Run SocMate from architecture through RTL pipeline in headless mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _project_root() -> Path:
    return Path(os.environ.get("SOCMATE_PROJECT_ROOT") or Path.cwd()).resolve()


def _load_requirements(path: str) -> str:
    return Path(path).read_text()


def _json_loads(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}
    return value if isinstance(value, dict) else {"value": value}


def _write_question_escalation(kind: str, state: dict) -> Path:
    esc_dir = _project_root() / ".socmate" / "escalations"
    esc_dir.mkdir(parents=True, exist_ok=True)
    path = esc_dir / f"{kind}.json"
    answers_path = esc_dir / f"{kind}.answers.json"
    payload = {
        "type": kind,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": state.get("phase"),
        "interrupt_type": state.get("interrupt_type"),
        "summary": state.get("interrupt_summary"),
        "ask_question": state.get("ask_question") or {},
        "answer_file": str(answers_path),
        "resume_action": "continue",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _question_answer_ids_from_escalation(payload: dict) -> list[str]:
    """Return all answer IDs the outer agent must cover for this escalation."""
    ask = payload.get("ask_question") or {}
    ids: list[str] = []
    seen: set[str] = set()
    for section in ("questions", "remaining_choice_questions", "auto_answerable"):
        for item in ask.get(section, []) or []:
            qid = str(item.get("id", "") or "").strip()
            if qid and qid not in seen:
                seen.add(qid)
                ids.append(qid)
    return ids


def _normalize_answer_payload(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    nested = value.get("answers")
    if isinstance(nested, dict):
        # Accept either the documented raw mapping or a defensive wrapper.
        return nested
    return value


def _validate_question_answers(answers: dict, escalation_payload: dict) -> tuple[list[str], list[str]]:
    """Return (missing_required_ids, extra_answer_ids)."""
    required = _question_answer_ids_from_escalation(escalation_payload)
    if not required:
        return [], []
    answer_ids = {str(k) for k in answers.keys()}
    missing = [qid for qid in required if qid not in answer_ids]
    extras = sorted(answer_ids - set(required))
    return missing, extras


def _recent_text(path: Path, max_lines: int = 160, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()[-max_lines:]
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"
    text = "\n".join(lines)
    return text[-max_chars:]


def _recent_context() -> dict:
    socmate = _project_root() / ".socmate"
    return {
        "pipeline_events_tail": _recent_text(socmate / "pipeline_events.jsonl"),
        "llm_calls_tail": _recent_text(socmate / "llm_calls.jsonl", max_lines=80),
        "run_log_tail": _recent_text(Path(os.environ.get("SOCMATE_RUN_LOG", "")), max_lines=240),
    }


def _write_decision_escalation(kind: str, state: dict, allowed_actions: list[str]) -> Path:
    esc_dir = _project_root() / ".socmate" / "escalations"
    esc_dir.mkdir(parents=True, exist_ok=True)
    path = esc_dir / f"{kind}.json"
    decision_path = esc_dir / f"{kind}.decision.json"
    if decision_path.exists():
        stale_path = esc_dir / (
            f"{kind}.decision.stale-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json"
        )
        decision_path.replace(stale_path)
    payload = {
        "type": kind,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "allowed_actions": allowed_actions,
        "decision_file": str(decision_path),
        "decision_schema": {
            "architecture": {
                "action": "accept | feedback | continue | abort",
                "feedback": "required for feedback; optional otherwise",
            },
            "pipeline": {
                "action": "approve | retry | skip | abort | fix_rtl | fix_tb",
                "block_actions": "optional object mapping block_name to action",
                "rationale": "required human/triage-agent rationale",
            },
        },
        "state": state,
        "recent_context": _recent_context(),
        "outer_agent_prompt": (
            "The outer agent must read this escalation plus .socmate OTEL/log "
            "artifacts, classify the root cause with cited evidence, and "
            "write an explicit decision JSON. Do not rubber-stamp retries."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


async def _wait_for_question_answers(
    kind: str,
    poll_s: float,
    escalation_path: Path | None = None,
) -> dict:
    esc_dir = _project_root() / ".socmate" / "escalations"
    answers_path = esc_dir / f"{kind}.answers.json"
    print(f"[arch] escalation pending: write answers JSON to {answers_path}", flush=True)
    while True:
        if answers_path.exists():
            raw_answers = json.loads(answers_path.read_text())
            answers = _normalize_answer_payload(raw_answers)
            if isinstance(answers, dict):
                if escalation_path and escalation_path.exists():
                    payload = json.loads(escalation_path.read_text())
                    missing, extras = _validate_question_answers(answers, payload)
                    if missing:
                        invalid_path = answers_path.with_name(
                            f"{answers_path.name}.invalid-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
                        )
                        answers_path.replace(invalid_path)
                        print(
                            "[arch] rejected incomplete escalation answers; "
                            f"missing {len(missing)} required id(s): {missing[:20]}. "
                            f"Moved invalid file to {invalid_path}",
                            flush=True,
                        )
                        await asyncio.sleep(max(5.0, poll_s))
                        continue
                    if extras:
                        print(
                            "[arch] escalation answers include extra context keys "
                            f"not in current question set: {extras[:20]}",
                            flush=True,
                        )
                print(f"[arch] loaded escalation answers from {answers_path}", flush=True)
                return answers
            raise RuntimeError(f"Escalation answers at {answers_path} must be a JSON object")
        await asyncio.sleep(max(5.0, poll_s))


async def _wait_for_decision(kind: str, poll_s: float, escalation_path: Path | None = None) -> dict:
    esc_dir = _project_root() / ".socmate" / "escalations"
    decision_path = esc_dir / f"{kind}.decision.json"
    if escalation_path:
        print(f"[headless] escalation written to {escalation_path}", flush=True)
    print(f"[headless] waiting for outer-agent decision JSON at {decision_path}", flush=True)
    while True:
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())
            if isinstance(decision, dict) and decision.get("action"):
                print(f"[headless] loaded decision from {decision_path}", flush=True)
                return decision
            raise RuntimeError(
                f"Decision at {decision_path} must be a JSON object with an action"
            )
        await asyncio.sleep(max(5.0, poll_s))


def _decision_feedback(decision: dict) -> str:
    """Return feedback text for architecture resume decisions.

    Triage agents often put their actionable text in ``rationale`` because the
    same schema is shared with pipeline decisions.  Architecture resumes require
    the explicit feedback parameter when action == feedback, so normalize here
    before calling the MCP tool.
    """
    return str(
        decision.get("feedback")
        or decision.get("rationale")
        or decision.get("root_cause")
        or ""
    )


def _frontend_integration_blocker(state: dict) -> str:
    """Return a reason if frontend integration is not backend-ready."""
    result = state.get("integration_result") or {}
    if not result:
        return ""
    if result.get("aborted"):
        return "integration check was aborted"
    if result.get("lint_clean") is False:
        return "integration top lint is not clean"
    try:
        error_count = int(result.get("error_count", 0) or 0)
    except (TypeError, ValueError):
        error_count = 0
    if error_count > 0:
        return f"integration check still reports {error_count} error(s)"
    return ""


def _answer_prd_questions(state: dict, requirements: str = "") -> dict:
    answers: dict[str, str] = {}
    ask = state.get("ask_question") or {}

    requirements_text = str(requirements or state.get("requirements", "") or "").lower()
    is_codec = any(token in requirements_text for token in ("h.264", "codec", "mort gif", "psnr", "bpp"))
    is_transformer = any(token in requirements_text for token in ("transformer", "llama2", "tinystories", "int4", "qspi"))

    rd_kpi_answer = (
        "Validation DV must preserve the Mort GIF RD targets from the prompt: "
        "software golden QP24 about 2.8674 bpp at PSNR >= 49.0 dB, QP36 "
        "about 1.0233 bpp at PSNR >= 38.0 dB, and QP48 about 0.2161 bpp at "
        "PSNR >= 34.0 dB. The hardware validation run should measure emitted "
        "bitstream size and reconstructed-frame PSNR against the golden model "
        "and fail if PSNR/bpp materially misses these targets."
    )
    transformer_kpi_answer = (
        "Validation DV must preserve the transformer accelerator KPIs from "
        "the prompt: bit-exact INT4/INT8 matmul over randomized tiles, "
        "bit-exact single-block checkpoint tensors within documented "
        "fixed-point tolerance, QSPI bandwidth/token-latency accounting, "
        "default on-chip SRAM budget no greater than 64 KB, and no on-chip "
        "weight storage beyond a prefetched tile plus metadata."
    )

    def _needs_default(value: object) -> bool:
        text = str(value or "").strip()
        if not text:
            return True
        lowered = text.lower()
        if lowered in {"n/a", "na", "none", "unknown", "tbd", "todo", "not specified"}:
            return True
        if lowered in {"n", "exact fixed-token sequence for n decoded tokens"}:
            return True
        return False

    def _maybe_answer_freeform_kpi(item: dict) -> None:
        qid = item.get("id")
        if not qid or qid in answers:
            return
        category = str(item.get("category", "")).lower()
        text = f"{item.get('question', '')} {item.get('context', '')}".lower()
        if is_codec and (
            "validation_kpi" in category
            or "kpi" in qid
            or "psnr" in text
            or "bpp" in text
            or "rate" in text and "distortion" in text
        ):
            answers[qid] = rd_kpi_answer
        elif is_transformer and (
            "validation_kpi" in category
            or "kpi" in qid
            or "matmul" in text
            or "checkpoint" in text
            or "flash" in text
            or "memory" in text
        ):
            answers[qid] = transformer_kpi_answer

    for item in ask.get("auto_answerable", []):
        qid = item.get("id")
        if qid:
            answers[qid] = item.get("suggested_answer", "")
        _maybe_answer_freeform_kpi(item)

    for item in ask.get("remaining_choice_questions", []):
        qid = item.get("id")
        opts = item.get("options") or []
        if qid and opts:
            answers[qid] = opts[0]
        _maybe_answer_freeform_kpi(item)

    for item in ask.get("questions", []):
        qid = item.get("id")
        opts = item.get("options") or []
        if qid and opts:
            answers[qid] = opts[0].get("label", "") if isinstance(opts[0], dict) else str(opts[0])
        _maybe_answer_freeform_kpi(item)

    common_defaults = {
        "target_technology": "SkyWater Sky130, sky130_fd_sc_hd, Verilog-2005",
        "target_clock": "50 MHz",
        "latency_budget": "No hard latency budget; prioritize lint/sim clean RTL and tractable area",
        "power_budget": "No explicit power budget; use synchronous single-clock RTL",
    }
    codec_defaults = {
        "bus_protocol": "AXI-Stream data interfaces with simple sideband mode pins",
        "data_width": "8-bit grayscale pixels, signed residuals, fp16 transform coefficients, int16 levels",
        "input_data_rate": "Mort GIF frames at the requirements resolution; scalable streaming raster input",
        "area_budget": "Fit a small soft-IP codec in Sky130; avoid SRAM-heavy or CPU-style designs",
        "validation_kpi": rd_kpi_answer,
        "rd_validation_kpi": rd_kpi_answer,
        "quality_kpi": rd_kpi_answer,
        "psnr_bpp_kpi": rd_kpi_answer,
    }
    transformer_defaults = {
        "primary_model_shape": "llama2.c TinyStories 260K-class: d_model=64, n_layers=5, n_heads=4, n_kv_heads=4, vocab_size=32000, max_seq_len=64, hidden_dim=172",
        "primary_model_dimensions": "llama2.c TinyStories 260K-class: d_model=64, n_layers=5, n_heads=4, n_kv_heads=4, vocab_size=32000, max_seq_len=64, hidden_dim=172",
        "flagship_model_shape": "llama2.c TinyStories 260K-class: d_model=64, n_layers=5, n_heads=4, n_kv_heads=4, vocab_size=32000, max_seq_len=64, hidden_dim=172",
        "exact_model_manifest": "Use a fixed synthetic 260K-class llama2.c manifest with d_model=64, n_layers=5, n_heads=4, n_kv_heads=4, hidden_dim=172, vocab_size=32000, max_seq_len=64; validation vectors may be generated from this manifest.",
        "fixed_point_formats": "INT4 weights are signed symmetric Q0.3, INT8 activations are signed Q3.4, RMSNorm/RoPE/SiLU LUT outputs are Q1.15, softmax probabilities are Q0.15, accumulators are INT32 with documented right-shift requantization.",
        "qspi_timing_parameters": "QSPI SDR at 50 MHz, 4 data lines, 8-bit command, 24-bit address, 8 dummy cycles, 256-byte bursts; use 15 MB/s sustained effective bandwidth for KPI accounting.",
        "flash_clock_and_protocol": "QSPI SDR at 50 MHz with 8-bit command, 24-bit address, 8 dummy cycles, 256-byte bursts, valid/ready backpressure, and underrun counters.",
        "end_to_end_decode_kpi": "Decode exactly 8 tokens from a fixed prompt and fixed synthetic model; every emitted token ID must match the golden reference.",
        "model_kpi_token_output": "Decode exactly 8 tokens from a fixed prompt and fixed synthetic model; every emitted token ID must match the golden reference.",
        "kpi_end_to_end_decode": "Decode exactly 8 tokens from a fixed prompt and fixed synthetic model; every emitted token ID must match the golden reference.",
        "decoded_token_count_kpi": "8 tokens",
        "end_to_end_token_count": "8 tokens",
        "bus_protocol": "Wishbone-compatible host control plus valid/ready tensor streams",
        "data_width": "INT4 weights, INT8 activations, INT32 accumulators, documented fixed-point tensor formats",
        "input_data_rate": "Host-sequenced token decode with external flash weight streaming",
        "area_budget": "Fit a Caravel-class Sky130 soft IP with <=64 KB default activation/KV SRAM and no bulk on-chip weights",
        "validation_kpi": transformer_kpi_answer,
        "matmul_bit_exact_kpi": "At least 64 randomized matrix tiles must match the fixed-point golden model exactly.",
        "single_block_bit_exact_kpi": "Golden checkpoint tensors for one transformer block must match within documented fixed-point tolerance.",
        "flash_bandwidth_model_kpi": "Bandwidth accounting must prove token-latency lower bounds from model size and flash throughput.",
        "onchip_memory_limit_kpi": "Default generated architecture must budget no more than 64 KB activation plus KV SRAM.",
        "no_onchip_weight_storage_kpi": "RTL must not allocate storage for more than one prefetched weight tile plus metadata.",
    }
    defaults = dict(common_defaults)
    if is_transformer:
        defaults.update(transformer_defaults)
    elif is_codec:
        defaults.update(codec_defaults)
    for key, value in defaults.items():
        if key not in answers or _needs_default(answers[key]):
            answers[key] = value

    return answers


async def _wait_task(task) -> None:
    if task is not None and not task.done():
        await asyncio.sleep(0)


async def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("SOCMATE_PROJECT_ROOT", str(Path.cwd()))
    os.environ.setdefault("SOCMATE_LLM_PROVIDER", "codex")
    os.environ.setdefault("SOCMATE_CODEX_MODEL", "gpt-5.5")
    os.environ.setdefault("SOCMATE_MODEL", "gpt-5.5")
    os.environ.setdefault("SOCMATE_BLOCK_MODEL", "gpt-5.5")
    os.environ.setdefault("SOCMATE_CODEX_SANDBOX", "danger-full-access")
    os.environ.setdefault("SOCMATE_SKIP_SYNTH", "1")

    # These stages are bypassed by default in architecture_graph.py. Leave the
    # enable env vars unset unless the caller explicitly exports them.
    os.environ.setdefault("SOCMATE_ENABLE_MEMORY_MAP", "0")
    os.environ.setdefault("SOCMATE_ENABLE_CLOCK_TREE", "0")
    os.environ.setdefault("SOCMATE_ENABLE_REGISTER_SPEC", "0")

    from orchestrator import mcp_server as mcp

    requirements = _load_requirements(args.requirements)

    if args.skip_architecture:
        print("[top] skipping architecture; using existing .socmate artifacts", flush=True)
    else:
        print("[top] starting architecture", flush=True)
        result = _json_loads(await mcp.start_architecture(
            requirements=requirements,
            target_clock_mhz=args.target_clock_mhz,
            pdk_config_path=args.pdk_config,
            max_rounds=args.max_rounds,
        ))
        print(json.dumps(result, indent=2), flush=True)
        if "error" in result:
            return 1

        last_arch_phase = ""
        while True:
            await asyncio.sleep(args.poll_s)
            await _wait_task(mcp._architecture.task)
            state = _json_loads(await mcp.get_architecture_state())
            phase = state.get("phase", "")
            if phase != last_arch_phase or state.get("human_input_needed"):
                print("[arch]", json.dumps({
                    "status": state.get("status"),
                    "phase": phase,
                    "blocks": state.get("block_names"),
                    "interrupt": state.get("interrupt_type"),
                    "summary": state.get("interrupt_summary"),
                }, indent=2), flush=True)
                last_arch_phase = phase

            if state.get("status") == "error":
                print(json.dumps(state, indent=2), flush=True)
                return 1
            if state.get("success") and state.get("status") == "done":
                break
            if state.get("human_input_needed"):
                itype = state.get("interrupt_type", "")
                if itype in ("prd_questions", "ers_questions"):
                    if args.auto_answer_questions:
                        answers = _answer_prd_questions(state, requirements)
                        print("[arch] auto-answering PRD questions", json.dumps(answers, indent=2), flush=True)
                    else:
                        path = _write_question_escalation(itype, state)
                        print(f"[arch] wrote question escalation to {path}", flush=True)
                        answers = await _wait_for_question_answers(itype, args.poll_s, path)
                    print(await mcp.resume_architecture("continue", json.dumps(answers)), flush=True)
                elif itype == "final_review":
                    payload_state = {
                        "phase": state.get("phase"),
                        "interrupt_type": itype,
                        "summary": state.get("interrupt_summary"),
                        "block_names": state.get("block_names"),
                        "architecture_state": state,
                    }
                    path = _write_decision_escalation(
                        "architecture_final_review", payload_state,
                        ["accept", "feedback", "abort"],
                    )
                    print(f"[arch] wrote final-review escalation to {path}", flush=True)
                    decision = await _wait_for_decision(
                        "architecture_final_review", args.poll_s, path
                    )
                    print(await mcp.resume_architecture(
                        decision.get("action", "feedback"),
                        _decision_feedback(decision),
                    ), flush=True)
                elif itype in (
                    "architecture_review_diagram",
                    "architecture_review_constraints",
                    "architecture_review_exhausted",
                    "architecture_review_needed",
                ):
                    payload_state = {
                        "phase": state.get("phase"),
                        "interrupt_type": itype,
                        "summary": state.get("interrupt_summary"),
                        "block_names": state.get("block_names"),
                        "architecture_state": state,
                    }
                    kind = f"architecture_{itype}"
                    path = _write_decision_escalation(
                        kind, payload_state, ["accept", "feedback", "abort"],
                    )
                    print(f"[arch] wrote architecture escalation to {path}", flush=True)
                    decision = await _wait_for_decision(kind, args.poll_s, path)
                    print(await mcp.resume_architecture(
                        decision.get("action", "feedback"),
                        _decision_feedback(decision),
                    ), flush=True)
                else:
                    payload_state = {
                        "phase": state.get("phase"),
                        "interrupt_type": itype,
                        "summary": state.get("interrupt_summary"),
                        "block_names": state.get("block_names"),
                        "architecture_state": state,
                    }
                    kind = f"architecture_{itype or 'unknown_interrupt'}"
                    path = _write_decision_escalation(
                        kind, payload_state, ["continue", "feedback", "abort"],
                    )
                    print(f"[arch] wrote unknown-interrupt escalation to {path}", flush=True)
                    decision = await _wait_for_decision(kind, args.poll_s, path)
                    print(await mcp.resume_architecture(
                        decision.get("action", "continue"),
                        _decision_feedback(decision),
                    ), flush=True)

    print("[top] architecture complete; starting frontend pipeline from block_specs.json", flush=True)
    result = _json_loads(await mcp.start_pipeline(
        max_attempts=args.max_attempts,
        target_clock_mhz=args.target_clock_mhz,
        blocks_file=args.blocks_file,
    ))
    print(json.dumps(result, indent=2), flush=True)
    if "error" in result:
        return 1

    last_pipeline = None
    retry_counts: dict[str, int] = {}
    while True:
        await asyncio.sleep(args.poll_s)
        await _wait_task(mcp._pipeline.task)
        state = _json_loads(await mcp.get_pipeline_state())
        digest = (
            state.get("status"),
            state.get("completed_count"),
            state.get("total_blocks"),
            state.get("current_tier"),
            state.get("interrupt_type"),
            state.get("pending_interrupt_count"),
            tuple(state.get("next_nodes") or []),
        )
        if digest != last_pipeline:
            print("[pipeline]", json.dumps({
                "status": state.get("status"),
                "completed": f"{state.get('completed_count')}/{state.get('total_blocks')}",
                "tier": state.get("current_tier"),
                "interrupt": state.get("interrupt_type"),
                "pending_interrupts": state.get("pending_interrupt_count"),
                "next_nodes": state.get("next_nodes") or [],
            }, indent=2), flush=True)
            last_pipeline = digest

        if state.get("status") == "error":
            print(json.dumps(state, indent=2), flush=True)
            return 1
        next_nodes = state.get("next_nodes") or []
        if state.get("status") == "done" and next_nodes:
            next_node = next_nodes[0]
            print(f"[pipeline] continuing pending graph node: {next_node}", flush=True)
            result = _json_loads(await mcp.restart_node(next_node))
            print(json.dumps(result, indent=2), flush=True)
            if "error" in result:
                return 1
            continue
        if state.get("pipeline_done") and not next_nodes:
            print("[top] pipeline finished", flush=True)
            print(json.dumps(state, indent=2), flush=True)
            blocker = _frontend_integration_blocker(state)
            if blocker:
                print(f"[top] frontend integration is not backend-ready: {blocker}", flush=True)
                return 1
            break
        if state.get("status") == "done" and not next_nodes:
            if state.get("completed_count") == state.get("total_blocks"):
                print("[top] pipeline finished", flush=True)
                print(json.dumps(state, indent=2), flush=True)
                blocker = _frontend_integration_blocker(state)
                if blocker:
                    print(
                        f"[top] frontend integration is not backend-ready: {blocker}",
                        flush=True,
                    )
                    return 1
                break
            print("[top] pipeline stopped before all blocks completed", flush=True)
            print(json.dumps(state, indent=2), flush=True)
            return 1
        if state.get("status") == "interrupted":
            payload = state.get("interrupt_payload") or {}
            if (
                isinstance(payload, dict)
                and payload.get("type") == "uarch_integration_review"
                and int(payload.get("issues_found", 0) or 0) == 0
                and not payload.get("review_failed")
            ):
                print("[pipeline] auto-approving clean uarch integration review", flush=True)
                print(await mcp.resume_pipeline(action="approve"), flush=True)
                continue
            actions = state.get("interrupt_actions") or []
            interrupted = state.get("interrupted_blocks") or []
            for item in interrupted:
                block = item.get("block_name", "")
                retry_counts[block] = retry_counts.get(block, 0) + 1
            payload_state = {
                "status": state.get("status"),
                "completed_count": state.get("completed_count"),
                "total_blocks": state.get("total_blocks"),
                "current_tier": state.get("current_tier"),
                "interrupt_type": state.get("interrupt_type"),
                "interrupt_actions": actions,
                "pending_interrupt_count": state.get("pending_interrupt_count"),
                "interrupted_blocks": interrupted,
                "retry_counts_seen_by_runner": retry_counts,
                "pipeline_state": state,
            }
            path = _write_decision_escalation(
                "pipeline_interrupt", payload_state,
                actions or ["retry", "abort"],
            )
            print(f"[pipeline] wrote interrupt escalation to {path}", flush=True)
            decision = await _wait_for_decision("pipeline_interrupt", args.poll_s, path)
            action = decision.get("action")
            block_actions = decision.get("block_actions") or {}
            print("[pipeline] applying decision", action, block_actions, flush=True)
            print(await mcp.resume_pipeline(
                action=action,
                block_actions=json.dumps(block_actions) if block_actions else "",
            ), flush=True)

    if not args.run_backend:
        return 0

    print("[top] starting backend", flush=True)
    result = _json_loads(await mcp.start_backend(
        max_attempts=args.max_attempts,
        target_clock_mhz=args.target_clock_mhz,
    ))
    print(json.dumps(result, indent=2), flush=True)
    if "error" in result:
        return 1

    last_backend = None
    while True:
        await asyncio.sleep(args.poll_s)
        await _wait_task(mcp._backend.task)
        state = _json_loads(await mcp.get_backend_state())
        digest = (
            state.get("status"),
            state.get("current_block"),
            state.get("phase"),
            state.get("attempt"),
            state.get("completed_count"),
            state.get("backend_done"),
            tuple(state.get("next_nodes") or []),
        )
        if digest != last_backend:
            print("[backend]", json.dumps({
                "status": state.get("status"),
                "current_block": state.get("current_block"),
                "phase": state.get("phase"),
                "attempt": state.get("attempt"),
                "completed": f"{state.get('completed_count')}/{state.get('total_blocks')}",
                "backend_done": state.get("backend_done"),
                "next_nodes": state.get("next_nodes") or [],
            }, indent=2), flush=True)
            last_backend = digest

        if state.get("status") == "error":
            print(json.dumps(state, indent=2), flush=True)
            return 1
        if state.get("backend_done") or state.get("status") == "done":
            print("[top] backend finished", flush=True)
            print(json.dumps(state, indent=2), flush=True)
            return 0
        if state.get("status") == "interrupted":
            payload_state = {
                "status": state.get("status"),
                "current_block": state.get("current_block"),
                "phase": state.get("phase"),
                "attempt": state.get("attempt"),
                "previous_error": state.get("previous_error"),
                "interrupt_payload": state.get("interrupt_payload"),
                "backend_state": state,
            }
            path = _write_decision_escalation(
                "backend_interrupt", payload_state, ["retry", "skip", "abort"],
            )
            print(f"[backend] wrote interrupt escalation to {path}", flush=True)
            decision = await _wait_for_decision("backend_interrupt", args.poll_s, path)
            print("[backend] applying decision", decision.get("action"), flush=True)
            print(await mcp.resume_backend(
                action=decision.get("action", "retry"),
                constraint=_decision_feedback(decision),
            ), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="examples/multiframe_codec_v2/requirements.md")
    parser.add_argument("--target-clock-mhz", type=float, default=50.0)
    parser.add_argument("--pdk-config", default="pdk/configs/sky130.yaml")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument(
        "--skip-architecture",
        action="store_true",
        help="Start from existing .socmate architecture artifacts and run frontend only.",
    )
    parser.add_argument(
        "--blocks-file",
        default="",
        help="Optional blocks.yaml path for frontend. Empty uses .socmate/block_specs.json.",
    )
    parser.add_argument(
        "--run-backend",
        action="store_true",
        help="Run backend after frontend completes.",
    )
    parser.add_argument(
        "--auto-answer-questions",
        action="store_true",
        default=os.environ.get("SOCMATE_HEADLESS_AUTO_ANSWER_QUESTIONS", "").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Use canned PRD/ERS answers. Default is to escalate questions and wait for an answers file.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
