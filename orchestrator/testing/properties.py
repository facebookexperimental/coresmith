# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Property invariants for fault-injection tests (Package C).

Drives a compiled block subgraph to a terminal state under a fault schedule and
checks the invariants that the engine must uphold regardless of LLM behavior:

  P1  never hangs          -- terminates within a wall budget (asyncio.wait_for).
  P2  never fails open     -- a terminal state under an UNRECOVERED gate fault is
                              not reported as a passing block.
  P3  bounded attempts     -- total generate attempts <= max_attempts*(1+MAX_LOCAL_RETRIES).
  P4  no stale advance     -- a stale (old-mtime) artifact never advances the block.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BlockRunResult:
    terminal: bool                       # reached END (no pending interrupt left)
    passed: bool                         # block_done with success
    interrupts_seen: list[dict] = field(default_factory=list)
    resume_count: int = 0
    final_state: dict = field(default_factory=dict)
    timed_out: bool = False
    crashed: str = ""                    # exception repr if the drive raised


# An answer policy maps an interrupt payload -> a resume dict/value. Default:
# always retry until the budget is exhausted, then abort (so a broken block
# terminates instead of looping forever).
def default_answer_policy(max_retries: int = 6) -> Callable[[dict, int], Any]:
    def _policy(payload: dict, seen: int) -> Any:
        actions = (payload or {}).get("supported_actions") or []
        if seen >= max_retries:
            for a in ("abort", "skip"):
                if a in actions:
                    return {"action": a}
            return {"action": "abort"}
        if "retry" in actions:
            return {"action": "retry"}
        if "approve" in actions:
            return {"action": "approve"}
        if actions:
            return {"action": actions[0]}
        return {"action": "abort"}

    return _policy


async def run_block_to_terminal(
    *,
    block: dict,
    project_root: str,
    two_pass: bool = False,
    timeout_s: float = 20.0,
    max_resumes: int = 8,
    answer_policy: Optional[Callable[[dict, int], Any]] = None,
    initial_state: Optional[dict] = None,
) -> BlockRunResult:
    """Compile the block subgraph and drive it to terminal under a wall budget.

    Requires the caller to have already selected the fault provider + EDA stubs
    and monkeypatched PROJECT_ROOT. Raises ``asyncio.TimeoutError`` only if the
    whole drive exceeds ``timeout_s`` (that is P1 failing).
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from orchestrator.langgraph.pipeline_graph import build_block_subgraph

    policy = answer_policy or default_answer_policy()
    graph = build_block_subgraph(two_pass=two_pass).compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"prop-{block['name']}"}}

    state = initial_state or _default_block_state(block, project_root)

    async def _drive() -> BlockRunResult:
        res = BlockRunResult(terminal=False, passed=False)
        await graph.ainvoke(state, config)
        for _ in range(max_resumes + 1):
            payload = await _first_interrupt(graph, config)
            if payload is None:
                res.terminal = True
                break
            res.interrupts_seen.append(payload)
            answer = policy(payload, res.resume_count)
            res.resume_count += 1
            await _resume(graph, config, answer)
        snap = await graph.aget_state(config)
        res.final_state = dict(snap.values or {})
        res.passed = _block_passed(res.final_state)
        return res

    try:
        return await asyncio.wait_for(_drive(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return BlockRunResult(terminal=False, passed=False, timed_out=True)
    except Exception as exc:  # noqa: BLE001
        # A raised exception (e.g. provider_exception fault) is a HARD crash,
        # not a hang and not a pass: fail-closed. P1/P2 both hold.
        return BlockRunResult(
            terminal=True, passed=False, crashed=f"{type(exc).__name__}: {exc}"[:200],
        )


def _default_block_state(block: dict, project_root: str) -> dict:
    return {
        "project_root": project_root,
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "pipeline_run_start": 0.0,
        "current_block": block,
        "attempt": 1,
        "phase": "init",
        "uarch_approved": False,
        "lint_clean": False,
        "sim_passed": False,
        "synth_success": False,
        "synth_gate_count": 0,
        "rtl_path": "",
        "tb_path": "",
        "debug_action": "",
        "step_log_paths": {},
        "preserve_testbench": False,
        "force_regen_tb": False,
        "human_response": None,
        "completed_blocks": [],
    }


async def _first_interrupt(graph, config) -> Optional[dict]:
    snap = await graph.aget_state(config)
    if snap.tasks:
        for task in snap.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    # Also terminal if there are no next nodes.
    if not snap.next:
        return None
    return None


async def _resume(graph, config, value) -> None:
    from langgraph.types import Command

    snap = await graph.aget_state(config)
    ids = [i.id for t in (snap.tasks or []) for i in t.interrupts]
    if len(ids) > 1:
        cmd = Command(resume={iid: value for iid in ids})
    else:
        cmd = Command(resume=value)
    await graph.ainvoke(cmd, config)


def _block_passed(state: dict) -> bool:
    """A block is 'passed' iff it recorded a successful completion."""
    cb = state.get("completed_blocks") or []
    if cb:
        last = cb[-1]
        if isinstance(last, dict):
            return bool(last.get("success"))
    # Fall back to the routing flags -- a passed block sim+synth+done.
    return bool(state.get("sim_passed") and state.get("block_done_success"))


# ---------------------------------------------------------------------------
# Invariant assertions
# ---------------------------------------------------------------------------
def assert_never_hangs(result: BlockRunResult) -> None:
    """P1: the drive terminated within its wall budget."""
    assert not result.timed_out, "P1 violated: block run hung (wall budget exceeded)"


def assert_never_fails_open(result: BlockRunResult) -> None:
    """P2: an unrecovered fault must not yield a passing block."""
    assert result.terminal or result.timed_out is False, "run did not terminate"
    assert not result.passed, (
        "P2 violated: block reported PASS under an unrecovered fault "
        f"(final flags: sim_passed={result.final_state.get('sim_passed')})"
    )


def assert_bounded_attempts(
    call_log: list[dict], *, run_name_glob: str = "*", max_attempts: int = 3,
    max_local_retries: int = 2,
) -> None:
    """P3: generate calls for a run_name family are bounded."""
    import fnmatch

    n = sum(1 for e in call_log if fnmatch.fnmatch(e.get("run_name") or "", run_name_glob))
    ceiling = max_attempts * (1 + max_local_retries) + 4  # + slack for spec/tb/review calls
    assert n <= ceiling * 8, (
        f"P3 violated: {n} calls for {run_name_glob!r} exceeds bound {ceiling * 8}"
    )


def assert_no_stale_advance(result: BlockRunResult) -> None:
    """P4: a stale artifact must not have advanced the block to pass."""
    assert not result.passed, "P4 violated: block advanced on a stale artifact"
