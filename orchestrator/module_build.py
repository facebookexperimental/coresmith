# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The one durable module-build contract shared by every public entry.

``coresmith build module <name>`` (daemon ``POST /build/module``), the
daemon's ``run restart-block`` compatibility route and the MCP
``restart_block`` tool all run :func:`start_build`; ``build resume`` /
``resume_build`` run :func:`resume_build`; ``build abort`` runs
:func:`abort_build`; ``run start`` and the MCP ``start_pipeline`` share the
pipeline-start refusal. The transports (HTTP, MCP) only translate the
returned record or the raised :class:`BuildRefusal`; the sequence, its
ordering and its invariants live here, once:

1. One operation at a time (the per-loop operation lock + the ``graph_launch`` lease): a start, a resume and an abort
   never interleave around their awaits, and nothing starts while the build
   lifecycle, the pipeline or the backend is driving the workspace.
2. The persisted run env is applied FIRST: readiness, the recorded identity
   and the execution all see the same authoritative configuration.
3. Refuse before anything else: the shared architecture stages must be done
   and still hold, the selected module must be ready
   (:func:`orchestrator.state_store.stages.module_ready`), no cluster-worker
   fan-out (its publications bypass the module graph), no other build of the
   module in flight, and an explicit seed must be the bound target's own
   source. Refusals happen before the tool preflight, before any binding is
   touched and before any checkpoint is reset.
4. Record the build (``builds`` row, status ``dispatched``) with the
   architecture-input identity and the resolved worker binding, *then*
   dispatch the existing block subgraph on a persistent checkpoint thread
   named after the build.
5. A resume re-validates the build's recorded inputs against the live
   project before the checkpoint is touched: a module whose spec, model,
   worker or tooling changed while it was parked gets ``BUILD_STALE``, never
   a mix of old provenance and new work.
6. The graph itself publishes success (``block_done_node`` commits the pass
   and the ``completed`` record together, after the same compatibility
   check); this module classifies the non-success terminal states of the
   lifecycle (parked, error, aborted) and recovers rows after a process
   restart -- a row dispatched but never checkpointed is a truthful start
   failure, not an in-flight build that blocks the module forever.

No HTTP, no FastAPI, no MCP objects here: callers pass their lifecycles.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
import time
import weakref
from pathlib import Path
from typing import Any, Callable

from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st

BUILD_GRAPH = "build"
BUILD_CHECKPOINT_DB = "build_checkpoint.db"

# Every launch -- a build start/resume/abort, a pipeline start/resume/
# continue/restart, a backend start/resume -- runs under this lock in the
# process, so two transports cannot interleave around their awaits; and under
# the ``graph_launch`` lease across processes (the daemon and an MCP server on
# the same project database), so two processes cannot either. A running
# build holds the ``graph:build`` lease for its duration, which every other
# process's launch checks. A graph lease is taken inside the section BEFORE
# the launch (a lease that cannot be taken refuses the launch; nothing runs
# unowned), bound to the exact task the launch started and released only
# with that task's token when that task ends: a replacement task's lease is
# never released by the retired task's cleanup. A dead holder's lease is
# stolen by pid liveness.
_OP_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _op_lock() -> asyncio.Lock:
    """The operation lock of the running event loop (an asyncio lock binds
    to the loop that first waits on it; one daemon process has one loop,
    a test process may have several)."""
    loop = asyncio.get_running_loop()
    lock = _OP_LOCKS.get(loop)
    if lock is None:
        lock = _OP_LOCKS[loop] = asyncio.Lock()
    return lock


# The task that holds the section: a launch that delegates to another
# launching entry in the same task (the daemon's backend routes delegate to
# the MCP implementation) is ONE operation and re-enters it; a task that was
# spawned from inside the section is not the owner and waits like any other.
_SECTION_OWNER: contextvars.ContextVar[str] = contextvars.ContextVar("coresmith_launch_section", default="")
LAUNCH_LEASE = "graph_launch"
LAUNCH_TTL_S = 120.0
BUILD_LEASE = "graph:build"
GRAPH_LEASES = {"build": BUILD_LEASE, "pipeline": "graph:pipeline", "backend": "graph:backend"}
GRAPH_LEASE_TTL_S = 7 * 24 * 3600.0
BUILD_LEASE_TTL_S = GRAPH_LEASE_TTL_S


class BuildRefusal(Exception):
    """A refused entry: ``code`` + ``blockers`` (stage-machine shaped) + ``hint``.
    ``http_status`` is advisory for HTTP transports (409 by default)."""

    def __init__(self, code: str, text: str, *, blockers: list[dict] | None = None, hint: str = "",
                 http_status: int = 409, **extra: Any):
        super().__init__(text)
        self.code = code
        self.text = text
        self.blockers = list(blockers or [])
        self.hint = hint
        self.http_status = http_status
        self.extra = extra

    def to_json(self) -> dict:
        return {"error": self.code, "message": self.text, "blocked_by": self.blockers, "hint": self.hint, **self.extra}


def _running(lifecycle) -> bool:
    task = getattr(lifecycle, "task", None)
    return task is not None and not task.done()


# ---------------------------------------------------------------- refusals
def cluster_fanout_refusal() -> BuildRefusal | None:
    """Cluster-worker fan-out publishes blocks through ``coresmith block-done``
    outside the module graph: not a qualified build. Refused for every
    qualifying entry; the migration is ``CORESMITH_FANOUT=block``."""
    from orchestrator.langgraph.pipeline_graph import cluster_fanout_enabled
    if cluster_fanout_enabled():
        return BuildRefusal(
            "CLUSTER_FANOUT_UNSUPPORTED",
            "CORESMITH_FANOUT=cluster dispatches native cluster workers whose block-done publications "
            "bypass the module graph; they cannot produce a recorded build",
            hint="set CORESMITH_FANOUT=block (the default) so every module runs through the block subgraph",
        )
    return None


def readiness_refusal(db, root, module: str) -> BuildRefusal | None:
    """Why ``module`` cannot be built now (None = ready). An engine primitive
    (the generated fabric) is buildable through the same subgraph -- its
    RTL, testbench and spec are generated by the build -- once the shared
    architecture stages hold."""
    specs = {b["name"]: b for b in db.block_specs()}
    if module not in specs:
        return BuildRefusal("UNKNOWN_MODULE", f"{module!r} is not a registered block",
                            hint="coresmith blocks lists the registered modules", http_status=404)
    shared = st.shared_ready(db, root)
    if shared:
        return BuildRefusal("ARCHITECTURE_NOT_READY", "the shared architecture stages are not done or no longer hold",
                            blockers=shared, hint="coresmith stage status; coresmith stage next")
    blockers = st.module_ready(db, root, module)
    if blockers:
        return BuildRefusal("MODULE_NOT_READY", f"module {module} is not ready to build", blockers=blockers,
                            hint="resolve every blocker (uArch spec, bound target, built reference model with its "
                                 "declared checks, worker binding), then build again")
    return cluster_fanout_refusal()


def pipeline_start_refusal(db, root) -> BuildRefusal | None:
    """Why the whole-SoC pipeline (``run start`` / MCP ``start_pipeline``)
    cannot fan out: the stage machine must have reached ``blocks`` with
    every earlier stage still holding -- no empty-stage exemption, no force
    bypass -- and cluster fan-out is refused. ``run start --force`` keeps
    only its meaning of replacing an existing run."""
    rows = {r["name"]: r for r in db.stage_rows()}
    if not rows:
        return BuildRefusal("STAGE_MACHINE_UNUSED", "the project has no stage rows: it starts at requirements",
                            blockers=[{"code": "STAGE_MACHINE_UNUSED", "text": "coresmith stage status", "ids": [], "count": 0}],
                            hint="register the requirements and advance the stage machine to blocks "
                                 "(coresmith stage next) before starting the pipeline")
    s = st.status(db, root)
    if s["index"] < st.STAGES.index("blocks"):
        return BuildRefusal("STAGE_BEFORE_BLOCKS", f"the stage machine is at {s['stage']}, before blocks",
                            blockers=s["blocked_by"], advisories=s["advisories"], stage=s["stage"],
                            hint="resolve the blockers and run coresmith stage next until the stage is blocks")
    regressed = st.regressed_stages(db, root, upto="blocks")
    if regressed:
        return BuildRefusal("STAGE_REGRESSED", "an earlier stage was marked done but no longer holds",
                            blockers=regressed, hint="coresmith stage status shows what changed")
    return cluster_fanout_refusal()


def validate_seed(db, root, module: str, seed_rtl: str) -> dict:
    """An explicit seed must be an existing file that IS one of the bound
    target's sources: the build never rebinds a target to point at a seed.
    Returns ``{path, sha256}``."""
    tgt = B.target_identity(root, module)
    if tgt is None:
        raise BuildRefusal("TARGET_UNBOUND", f"module {module} has no bound target; a seed is one of the target's sources",
                           hint="coresmith target bind <module> --file <json> (sources include the seed)")
    p = Path(seed_rtl)
    if not p.is_absolute():
        p = Path(root) / p
    p = p.resolve()
    if not p.is_file():
        raise BuildRefusal("SEED_MISSING", f"seed RTL {p} does not exist", http_status=400)
    sources = {str(Path(s).resolve()) for s in (tgt.get("sources") or [])}
    if str(p) not in sources:
        raise BuildRefusal("SEED_NOT_BOUND", f"seed RTL {p} is not a source of the bound target {module}",
                           hint="bind the target to the seed first (coresmith target bind) -- the build does not "
                                "rebind targets", http_status=400)
    return {"path": str(p), "sha256": B.file_sha256(p)}


def in_flight_refusal(db, module: str) -> BuildRefusal | None:
    active = B.active_builds(db, module)
    if active:
        b = active[0]
        return BuildRefusal("BUILD_IN_FLIGHT", f"build {b['id']} of {module} is {b['status']}",
                            build_id=b["id"], status=b["status"],
                            hint="resume it (coresmith build resume --build-id ...), wait for it to finish, or abort "
                                 "it explicitly (coresmith build abort --build-id ...); a second start never "
                                 "overwrites an in-flight build")
    return None


def busy_refusal(lifecycle, busy: list | None = None) -> BuildRefusal | None:
    """Nothing starts or resumes while a graph is driving the workspace."""
    if _running(lifecycle):
        return BuildRefusal("BUILD_RUNNING", f"a module build is running (thread {lifecycle.thread_id}); one at a time",
                            hint="wait for it to park or finish, or coresmith build pause")
    for other in busy or []:
        if _running(other):
            name = getattr(other, "name", "graph")
            return BuildRefusal(f"{name.upper()}_RUNNING", f"the {name} graph is running; builds and the {name} share "
                                "the workspace -- pause it first")
    return None


def _live_lease(db, name: str) -> dict | None:
    """The lease row when it is held by a live process (not expired, holder
    alive when on this host), else None."""
    if not hasattr(db, "lease"):
        return None
    row = db.lease(name)
    if not row:
        return None
    if float(row.get("expires_ts") or 0) < time.time():
        return None
    try:
        from orchestrator.state_store.leases import hostname, pid_is_alive
        if row.get("holder_host") == hostname() and not pid_is_alive(int(row.get("holder_pid") or 0)):
            return None
    except Exception:  # noqa: BLE001
        pass
    return row


def graph_running_elsewhere(db, graph: str = "build") -> dict | None:
    """The ``graph:<graph>`` lease when another live process holds it."""
    row = _live_lease(db, GRAPH_LEASES[graph])
    if row and int(row.get("holder_pid") or 0) != os.getpid():
        return row
    return None


def build_running_elsewhere(db) -> dict | None:
    """The ``graph:build`` lease when another live process holds it."""
    return graph_running_elsewhere(db, "build")


def foreign_graph_refusal(db) -> BuildRefusal | None:
    """A build, the pipeline or the backend running in ANOTHER live process
    (its ``graph:<name>`` lease): every launch here is refused."""
    for graph, code in (("build", "BUILD_RUNNING"), ("pipeline", "PIPELINE_RUNNING"), ("backend", "BACKEND_RUNNING")):
        other = graph_running_elsewhere(db, graph)
        if other:
            meta = other.get("meta") if isinstance(other.get("meta"), dict) else {}
            what = f"build {meta.get('build_id')}" if graph == "build" else f"thread {meta.get('thread_id')}"
            return BuildRefusal(code, f"the {graph} graph is running in another process (pid "
                                f"{other.get('holder_pid')} on {other.get('holder_host')}, {what}); wait for it "
                                "to park or finish")
    return None


@contextlib.asynccontextmanager
async def launch_section(db, *, what: str):
    """The one critical section every launch runs in: the operation lock
    (this process) and the ``graph_launch`` lease (every process on this
    project), from the conflict checks to the start that follows them. A
    graph running in another process refuses the launch. Re-entrant for the
    task that already holds it (a route delegating to another launching
    entry is one operation); a task spawned from inside is not the owner."""
    me = f"{id(asyncio.current_task())}:{what}"
    owner = _SECTION_OWNER.get()
    if owner and owner.split(":", 1)[0] == str(id(asyncio.current_task())):
        yield
        return
    async with _op_lock():
        token = None
        if hasattr(db, "acquire_lease"):
            token = db.acquire_lease(LAUNCH_LEASE, LAUNCH_TTL_S, meta={"what": what})
            if token is None:
                row = db.lease(LAUNCH_LEASE) or {}
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                raise BuildRefusal("WORKSPACE_BUSY", f"another process is launching ({meta.get('what') or 'a graph'}, "
                                   f"pid {row.get('holder_pid')} on {row.get('holder_host')}); retry",
                                   hint="coresmith leases shows the holder")
        reset = _SECTION_OWNER.set(me)
        try:
            refusal = foreign_graph_refusal(db)
            if refusal:
                raise refusal
            yield
        finally:
            _SECTION_OWNER.reset(reset)
            if token:
                db.release_lease(LAUNCH_LEASE, token)


# The graph leases the current task chain owns (``{lease name: token}``): a
# launch that delegates to another launching entry in the same task (the
# daemon's backend routes to the MCP implementation) reuses the ownership
# instead of taking the lease twice.
_HELD_LEASES: contextvars.ContextVar[dict] = contextvars.ContextVar("coresmith_graph_leases", default={})


def _acquire_graph_lease(db, name: str, meta: dict | None = None) -> str:
    """Take ``name`` for this process. A lease this process left behind (its
    task gone) is superseded; a live foreign holder was refused by the
    launch section before this point; a lease that cannot be taken or
    persisted is an error the caller must act on -- never a running graph
    nobody owns."""
    row = db.lease(name)
    if row and int(row.get("holder_pid") or 0) == os.getpid():
        db.steal_lease(name, "superseded by a new task in the same process")
    token = db.acquire_lease(name, GRAPH_LEASE_TTL_S, meta=meta or {})
    if token is None:
        row = db.lease(name) or {}
        raise BuildRefusal("GRAPH_LEASE_UNAVAILABLE", f"{name} could not be taken (held by pid {row.get('holder_pid')} "
                           f"on {row.get('holder_host')}); nothing was launched", http_status=500)
    return token


def _record_ownership(lifecycle, name: str, token: str, task) -> None:
    """The lifecycle's lease record: exactly this token for exactly this task."""
    lifecycle._graph_lease = {"name": name, "token": token, "task": task}


def _release_ownership(db, lifecycle, name: str, token: str) -> None:
    """Release exactly ``token`` (the database release is token-scoped: a
    token another task superseded releases nothing) and clear the
    lifecycle's record only while it still refers to this token."""
    if hasattr(db, "release_lease"):
        try:
            db.release_lease(name, token)
        except Exception:  # noqa: BLE001
            pass
    rec = getattr(lifecycle, "_graph_lease", None) if lifecycle is not None else None
    if rec and rec.get("token") == token:
        lifecycle._graph_lease = None


async def _release_after(db, lifecycle, name: str, token: str, task) -> None:
    """Release ``token`` when ``task`` -- the exact task it was taken for --
    ends; a replacement task that took its own lease is untouched."""
    try:
        await asyncio.shield(task)
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - the task records its own outcome
        pass
    _release_ownership(db, lifecycle, name, token)


def hold_graph_lease(db, lifecycle, graph: str, *, meta: dict | None = None) -> str | None:
    """Hold ``graph:<graph>`` for the task ``lifecycle`` is running NOW, so
    another process's launch sees it (``foreign_graph_refusal``), and release
    it when exactly that task ends. The lifecycle's record already naming
    this task (and the database holding that token) is reused, not taken
    again. Returns the token (None when nothing runs)."""
    if not hasattr(db, "acquire_lease") or not _running(lifecycle):
        return None
    name = GRAPH_LEASES[graph]
    task = lifecycle.task
    rec = getattr(lifecycle, "_graph_lease", None)
    if rec and rec.get("name") == name and rec.get("task") is task:
        row = db.lease(name)
        if row and row.get("token") == rec.get("token") and int(row.get("holder_pid") or 0) == os.getpid():
            return rec["token"]
    token = _acquire_graph_lease(db, name, {**(meta or {}), "thread_id": getattr(lifecycle, "thread_id", "")})
    _record_ownership(lifecycle, name, token, task)
    asyncio.create_task(_release_after(db, lifecycle, name, token, task))
    return token


@contextlib.asynccontextmanager
async def graph_ownership(db, graph: str, lifecycle_getter: Callable[[], Any], *, meta: dict | None = None):
    """Own ``graph:<graph>`` across a launch: taken BEFORE the launch (inside
    the launch section, after the foreign-holder check), so a lease that
    cannot be taken refuses the launch instead of leaving a running graph
    nobody owns; after the launch, bound to the exact task the lifecycle is
    running (released when that task ends) or released at once when nothing
    started. A nested launch in the same task chain reuses the ownership."""
    name = GRAPH_LEASES[graph]
    held = _HELD_LEASES.get()
    if name in held or not hasattr(db, "acquire_lease"):
        yield
        return
    token = _acquire_graph_lease(db, name, {**(meta or {}), "graph": graph})
    reset = _HELD_LEASES.set({**held, name: token})
    bound = False
    lifecycle = None
    try:
        yield
        try:
            lifecycle = lifecycle_getter()
        except Exception:  # noqa: BLE001
            lifecycle = None
        task = getattr(lifecycle, "task", None)
        if lifecycle is not None and task is not None and not task.done():
            _record_ownership(lifecycle, name, token, task)
            asyncio.create_task(_release_after(db, lifecycle, name, token, task))
            bound = True
    finally:
        _HELD_LEASES.reset(reset)
        if not bound:
            _release_ownership(db, lifecycle, name, token)


def _build_ownership(db, lifecycle, build_id: str):
    """``graph:build`` across a build launch: taken before the start, bound
    to the started task (released when it ends, by :func:`watch_build` or
    the task's own watcher), released at once when the start failed."""
    return graph_ownership(db, "build", lambda: lifecycle,
                           meta={"build_id": build_id, "thread_id": getattr(lifecycle, "thread_id", "")})


def _release_build_lease(db, lifecycle, task=None) -> None:
    """Release the lifecycle's recorded build lease -- only when the record
    still names ``task`` (the task the caller watched)."""
    rec = getattr(lifecycle, "_graph_lease", None)
    if not rec:
        return
    if task is not None and rec.get("task") is not task:
        return
    _release_ownership(db, lifecycle, rec["name"], rec["token"])


# ---------------------------------------------------------------- planning
# Implementation attempts per build: a target miss, a failed simulation or
# synthesis goes back to the worker until they are used up (1 = no retry).
DEFAULT_MAX_ATTEMPTS = 3


def target_refusal(db, root, module: str) -> BuildRefusal | None:
    """Why the module's bound measurements cannot run
    in THIS process's environment (``FRD_TARGET_UNMEASURABLE``: area needs the
    liberty-mapped synthesis, power also OpenSTA). The structural target
    problems are module readiness (``MODULE_NOT_READY``)."""
    from orchestrator.state_store import module_targets as MT
    alloc = MT.allocation(db, root, module)
    try:
        from orchestrator.langgraph.pipeline_helpers import LIBERTY_FILE, _sta_problem
        liberty_present, sta_problem = Path(LIBERTY_FILE).is_file(), _sta_problem()
    except Exception as exc:  # noqa: BLE001
        liberty_present, sta_problem = False, f"engine tooling unavailable: {type(exc).__name__}"
    generic = (os.environ.get("CORESMITH_SYNTH_GENERIC", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    unmeasurable = MT.measurability(alloc, liberty_present=liberty_present, synth_generic=generic,
                                    sta_problem=sta_problem)
    if unmeasurable:
        return BuildRefusal(
            "FRD_TARGET_UNMEASURABLE", "bound targets cannot be measured in this environment; a build would only park",
            blockers=[{"code": p["code"], "text": p["text"], "ids": [p["where"]], "count": 1} for p in unmeasurable],
            hint="install / point the engine at the tool (PDK liberty, OpenSTA: CORESMITH_REAL_STA) or rebind the "
                 "target (coresmith frd verifier)")
    return None


def plan_module_build(db, root, module: str, *, entry: str, seed_rtl: str = "",
                      max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                      target_clock_mhz: float | None = None, uarch_feedback: str = "") -> dict:
    """Run every refusal, then compute the build's identity and initial
    graph state. Nothing is written: the caller runs its tool preflight and
    then :func:`dispatch`. Raises :class:`BuildRefusal`."""
    from orchestrator.langgraph.pipeline_helpers import resolve_run_clock_mhz
    if uarch_feedback.strip():
        raise BuildRefusal("SPEC_REVISION_REQUIRED", "module builds consume the registered uArch spec unchanged",
                           hint="edit the spec, register it with coresmith register uarch --block <module> <path>, "
                                "then start a new build", http_status=400)
    if int(max_attempts) < 1:
        raise BuildRefusal("BAD_MAX_ATTEMPTS", f"--max-attempts must be >= 1 (got {max_attempts})", http_status=400)
    refusal = readiness_refusal(db, root, module) or in_flight_refusal(db, module)
    if refusal:
        raise refusal
    spec = next(b for b in db.block_specs() if b["name"] == module)
    primitive = B.is_primitive_spec(spec)
    if seed_rtl and primitive:
        raise BuildRefusal("SEED_NOT_ALLOWED", f"{module} is an engine primitive: its RTL is generated, never seeded",
                           http_status=400)
    refusal = target_refusal(db, root, module)
    if refusal:
        raise refusal
    seed = validate_seed(db, root, module, seed_rtl) if seed_rtl else None
    clock = resolve_run_clock_mhz(target_clock_mhz, root)
    # this process runs the build's tools: their identity is its own
    # resolution, recorded for every other process (builds.tooling_snapshot)
    inputs = B.module_inputs(db, root, module, seed_path=seed["path"] if seed else None, target_clock_mhz=clock,
                             persist_tools=True)
    build_id = B.new_build_id(module)
    thread_id = f"build-{build_id}"
    return {"build_id": build_id, "thread_id": thread_id, "module": module, "entry": entry, "seed": seed,
            "inputs": inputs, "worker": inputs["worker"], "target_clock_mhz": clock, "primitive": primitive,
            "max_attempts": int(max_attempts), "uarch_feedback": uarch_feedback or "", "block_spec": spec}


def targets_summary(inputs: dict) -> dict:
    """What a build binds, for the start response and ``build show``."""
    t = inputs.get("targets") or {}
    acc = inputs.get("acceptance") or None
    return {"required": [x["id"] for x in t.get("targets") or [] if x.get("required")],
            "advisory": [x["id"] for x in t.get("targets") or [] if not x.get("required")],
            "functional": [x["id"] for x in t.get("functional") or []],
            "deferred": list(t.get("deferred") or []), "unverified": list(t.get("unverified") or []),
            "digest": t.get("digest"), "frd_revision": t.get("frd_revision"),
            "acceptance_tb": (acc or {}).get("path"),
            "acceptance_files": [f.get("path") for f in (acc or {}).get("files") or []] or
                                ([acc.get("path")] if acc else [])}


def initial_block_state(root, plan: dict) -> dict:
    """The ``BlockState`` the block subgraph starts from: the registered spec
    is implemented as-is (``reuse_spec``) and checked for feasibility; an explicit seed
    is consumed by the first RTL pass; a primitive keeps its generated
    testbench."""
    return {
        "project_root": str(root),
        "target_clock_mhz": plan["target_clock_mhz"],
        "max_attempts": plan["max_attempts"],
        "pipeline_run_start": time.time(),
        "current_block": dict(plan["block_spec"]),
        "attempt": 1,
        "phase": "init",
        "build_id": plan["build_id"],
        "seed_rtl": bool(plan.get("seed")),
        "uarch_approved": False,
        "uarch_feedback": plan.get("uarch_feedback") or "",
        "reuse_spec": True,
        "lint_clean": False,
        "sim_passed": False,
        "synth_success": False,
        "synth_gate_count": 0,
        "rtl_path": "",
        "tb_path": "",
        "debug_action": "",
        "human_response": None,
        "completed_blocks": [],
        "step_log_paths": {},
        "preserve_testbench": bool(plan.get("primitive")),
        "force_regen_tb": False,
        "abort_reason": "",
    }


def dispatch(db, root, plan: dict) -> dict:
    """Record the build's immutable input identity before the graph starts."""
    row = B.record_dispatch(db, build_id=plan["build_id"], module=plan["module"], entry=plan["entry"],
                            graph=BUILD_GRAPH, thread_id=plan["thread_id"], inputs=plan["inputs"],
                            worker=plan["worker"], seed=plan.get("seed"))
    return row


# ---------------------------------------------------------------- the shared sequences
async def start_build(db, root, lifecycle, *, module: str, entry: str, busy: list | None = None,
                      seed_rtl: str = "", max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                      target_clock_mhz: float | None = None,
                      uarch_feedback: str = "", apply_env: Callable[[], Any] | None = None,
                      preflight: Callable[[], Any] | None = None) -> dict:
    """The one start sequence (see the module docstring). ``apply_env``
    re-reads the persisted run env (applied before readiness so the
    recorded identity and the execution agree); ``preflight`` is the
    transport's tool check and runs only after every refusal passed. Raises
    :class:`BuildRefusal`; returns the started record. The caller schedules
    :func:`watch_build` for the returned ``build_id``."""
    from orchestrator.state_store.store import Scoreboard
    from orchestrator.state_store.trust import capture_run_baseline
    async with launch_section(db, what=f"build module {module}"):
        refusal = busy_refusal(lifecycle, busy)
        if refusal:
            raise refusal
        env_updated = list(apply_env() or []) if apply_env else []
        plan = plan_module_build(db, root, module, entry=entry, seed_rtl=seed_rtl, max_attempts=max_attempts,
                                 target_clock_mhz=target_clock_mhz, uarch_feedback=uarch_feedback)
        # Every refusal has passed: now the tooling and the recorded state.
        if preflight:
            preflight()
        if not Scoreboard(root).ensure_schema():
            raise BuildRefusal("SCHEMA_INIT_FAILED", "the measurement tables could not be initialised; a build cannot "
                               "be recorded", http_status=500)
        try:
            capture_run_baseline(root)
        except RuntimeError as exc:
            raise BuildRefusal("TRUST_BASELINE_FAILED", str(exc), http_status=500) from exc
        try:
            await lifecycle.select_thread(plan["thread_id"])
        except RuntimeError as exc:
            raise BuildRefusal("BUILD_RUNNING", str(exc)) from exc
        async with _build_ownership(db, lifecycle, plan["build_id"]):
            try:
                dispatch(db, root, plan)
            except Exception as exc:  # noqa: BLE001 - an unrecorded build is not started
                raise BuildRefusal("BUILD_RECORD_FAILED", f"the build could not be recorded: {exc}",
                                   http_status=500) from exc
            try:
                await lifecycle.safe_start(initial_block_state(root, plan), _config(plan["thread_id"]))
            except RuntimeError as exc:
                B.mark_status(db, plan["build_id"], "error", error=f"the graph did not start: {exc}"[:1000])
                raise BuildRefusal("BUILD_START_FAILED", str(exc), build_id=plan["build_id"]) from exc
        B.mark_started(db, plan["build_id"], thread_id=plan["thread_id"])
    out = {"started": True, "build_id": plan["build_id"], "thread_id": plan["thread_id"], "module": plan["module"],
           "entry": entry, "status": "running", "inputs_digest": plan["inputs"]["digest"],
           "worker": plan["worker"], "seed": plan.get("seed"), "lifecycle_status": lifecycle.status,
           "targets": targets_summary(plan["inputs"]), "max_attempts": plan["max_attempts"],
           "retries": max(0, plan["max_attempts"] - 1)}
    if env_updated:
        out["env_updated"] = env_updated
    return out


def stale_refusal(db, root, row: dict, *, allow_axes: set[str] | frozenset = frozenset()) -> BuildRefusal | None:
    """``BUILD_STALE`` when the live project no longer matches the inputs the
    build recorded (its spec, contracts, target configuration, model,
    harness, owned items, FRD targets, acceptance testbench, worker binding,
    tooling). ``allow_axes``: axes the answer to the current park itself
    resolves (the ``acceptance_changed`` park's ``restore`` / ``abort``)."""
    stored = row.get("inputs") or {}
    live = B.module_inputs(db, root, row["module"], seed_path=(stored.get("seed") or {}).get("path"),
                           target_clock_mhz=(stored.get("tooling") or {}).get("clock_mhz"))
    stale = [r for r in B.stale_reasons(stored, live) if r.split(":", 1)[0] not in allow_axes]
    if stale:
        return BuildRefusal("BUILD_STALE", f"build {row['id']} recorded inputs the project no longer has",
                            blockers=[{"code": "INPUT_CHANGED", "text": r, "ids": [], "count": 0} for r in stale],
                            hint="start a new build (coresmith build module); the parked one keeps its history",
                            build_id=row["id"])
    return None


async def resume_build(db, root, lifecycle, build_id: str, *, action: str, busy: list | None = None,
                       feedback: str = "", rtl_fix_description: str = "", rationale: str = "",
                       interrupt_id: str | None = None, actor: str = "cli",
                       apply_env: Callable[[], Any] | None = None,
                       on_parks: Callable[[list[dict]], Any] | None = None) -> dict:
    """The one resume sequence: the build row is selected from its persisted
    identity, the recorded inputs are re-validated against the live project,
    and only then is the checkpoint read and the park answered with ONE
    action per its ``supported_actions``. A build stopped at a node boundary
    with nothing parked is ticked (``ticked``). Raises :class:`BuildRefusal`.
    The caller schedules :func:`watch_build`."""
    from langgraph.types import Command
    async with launch_section(db, what=f"build resume {build_id}"):
        row = B.get_build(db, build_id)
        if row is None:
            raise BuildRefusal("UNKNOWN_BUILD", f"no build {build_id}", http_status=404)
        if row["status"] in B.BUILD_TERMINAL:
            raise BuildRefusal("BUILD_TERMINAL", f"build {build_id} is {row['status']}; start a new build")
        if row.get("graph") != BUILD_GRAPH:
            raise BuildRefusal("NOT_A_MODULE_BUILD", f"build {build_id} was dispatched by the {row.get('graph')} graph; "
                               "resume that graph instead (coresmith resume)")
        refusal = busy_refusal(lifecycle, busy)
        if refusal:
            raise refusal
        env_updated = list(apply_env() or []) if apply_env else []
        refusal = stale_refusal(db, root, row)
        # The one stale input a park answers itself: the acceptance testbench
        # changed during the build and the build is parked at
        # ``acceptance_changed`` -- ``restore`` puts the recorded oracle back,
        # ``abort`` ends the build. Nothing else may differ.
        acceptance_only = refusal is not None and action in ("restore", "abort") and \
            stale_refusal(db, root, row, allow_axes={"acceptance"}) is None
        if refusal and not acceptance_only:
            raise refusal
        try:
            await lifecycle.select_thread(row["thread_id"])
        except RuntimeError as exc:
            raise BuildRefusal("BUILD_RUNNING", str(exc)) from exc
        config = _config(row["thread_id"])
        parks, next_nodes = await live_parks(lifecycle, row["thread_id"])
        if acceptance_only and not (parks and all(p["payload"].get("type") == "acceptance_changed" for p in parks)):
            raise refusal
        if on_parks:
            on_parks(parks)
        if interrupt_id:
            parks = [p for p in parks if p["interrupt_id"] == interrupt_id]
            if not parks:
                raise BuildRefusal("NO_SUCH_INTERRUPT", f"no parked interrupt {interrupt_id!r} in build {build_id}",
                                   http_status=404)
        if not parks:
            if not next_nodes:
                raise BuildRefusal("NOTHING_TO_RESUME", "no pending interrupt and nothing scheduled in this build's "
                                   "checkpoint")
            async with _build_ownership(db, lifecycle, build_id):
                B.mark_status(db, build_id, "running", terminal=False)
                await lifecycle.safe_start(None, config)
            out = {"resumed": True, "ticked": True, "build_id": build_id, "next_nodes": next_nodes,
                   "status": lifecycle.status, "answered": []}
            if env_updated:
                out["env_updated"] = env_updated
            return out
        for p in parks:
            allowed = p["payload"].get("supported_actions") or []
            if allowed and action not in allowed:
                raise BuildRefusal("ACTION_UNSUPPORTED", f"action '{action}' not supported by the parked interrupt",
                                   allowed=allowed, http_status=400)
        value = resume_value(action, feedback=feedback, rtl_fix_description=rtl_fix_description,
                             rationale=rationale, actor=actor)
        cmd = Command(resume={p["lg_id"]: value for p in parks}) if (len(parks) > 1 or interrupt_id) else Command(resume=value)
        for p in parks:
            if p["interrupt_id"]:
                try:
                    db.resolve_interrupt(p["interrupt_id"], value, resolved_by=actor or "cli")
                except Exception:  # noqa: BLE001 - bookkeeping; the checkpoint is the authority
                    pass
        try:
            db.consume_lg_interrupts([p["lg_id"] for p in parks])
        except Exception:  # noqa: BLE001
            pass
        async with _build_ownership(db, lifecycle, build_id):
            B.mark_status(db, build_id, "running", terminal=False)
            await lifecycle.safe_resume(cmd, config)
    out = {"resumed": True, "build_id": build_id, "interrupts": len(parks), "action": action,
           "status": lifecycle.status, "answered": [p["payload"] for p in parks]}
    if env_updated:
        out["env_updated"] = env_updated
    return out


async def abort_build(db, lifecycle, build_id: str, *, reason: str = "", actor: str = "cli") -> dict:
    """Mark a build that is not running ``aborted`` (history retained, nothing
    published) so the module can be built again. A build whose task is
    running must be paused first."""
    async with _op_lock():
        row = B.get_build(db, build_id)
        if row is None:
            raise BuildRefusal("UNKNOWN_BUILD", f"no build {build_id}", http_status=404)
        if row["status"] in B.BUILD_TERMINAL:
            raise BuildRefusal("BUILD_TERMINAL", f"build {build_id} is already {row['status']}")
        if _running(lifecycle) and lifecycle.thread_id == row["thread_id"]:
            raise BuildRefusal("BUILD_RUNNING", f"build {build_id} is running; coresmith build pause first")
        other = build_running_elsewhere(db)
        if other and isinstance(other.get("meta"), dict) and other["meta"].get("build_id") == build_id:
            raise BuildRefusal("BUILD_RUNNING", f"build {build_id} is running in another process (pid "
                               f"{other.get('holder_pid')}); pause it there first")
        B.mark_status(db, build_id, "aborted", error=f"aborted by {actor or 'cli'}" + (f": {reason}" if reason else ""))
        # One build of a module is active at a time, so the module's open
        # build-graph parks are this build's: nothing will answer them now.
        abandoned: list[str] = []
        if row.get("graph") == BUILD_GRAPH and hasattr(db, "abandon_parks"):
            try:
                abandoned = db.abandon_parks(graph=BUILD_GRAPH, block=row["module"])
            except Exception:  # noqa: BLE001 - bookkeeping; the build is aborted either way
                abandoned = []
    return {"aborted": True, "build_id": build_id, "status": "aborted", "module": row["module"],
            "abandoned_interrupts": abandoned}


def abort_pipeline_builds(db, *, reason: str) -> list[str]:
    """Every active row the PIPELINE graph dispatched is aborted (``run start
    --force`` discards that run's checkpoint; its in-flight builds cannot
    complete). Returns the aborted ids."""
    out = []
    for row in B.active_builds(db):
        if row.get("graph") == "pipeline":
            B.mark_status(db, row["id"], "aborted", error=reason[:500])
            out.append(row["id"])
    return out


# ---------------------------------------------------------------- lifecycle
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def live_parks(lifecycle, thread_id: str) -> tuple[list[dict], list[str]]:
    """``([{lg_id, interrupt_id, payload}], next_nodes)`` of a build thread's
    checkpoint (empty while its task runs)."""
    await lifecycle.ensure_graph()
    snap = await lifecycle.graph.aget_state(_config(thread_id))
    parks = []
    for t in (snap.tasks if snap and snap.tasks else []):
        for i in t.interrupts:
            val = i.value if isinstance(i.value, dict) else {"value": i.value}
            parks.append({"lg_id": i.id, "interrupt_id": str(val.get("interrupt_id") or ""), "payload": val})
    return parks, (list(snap.next) if snap and snap.next else [])


async def finalize_after_task(db, root, lifecycle, build_id: str) -> str:
    """Classify a finished lifecycle task for ``build_id``. ``completed`` is
    written only by ``block_done_node`` (with the published pass, in one
    transaction); everything else is recorded here: ``parked`` (an interrupt
    awaits the Architect), ``aborted`` (the graph ended without a success and
    the block result says abort/skip), ``failed`` (the block result failed),
    ``error`` (the task errored or ended without a block result). Idempotent:
    a build already terminal is left alone."""
    row = B.get_build(db, build_id)
    if row is None:
        return "unknown"
    if row["status"] in B.BUILD_TERMINAL:
        return row["status"]
    status = lifecycle.status
    try:
        parks, next_nodes = await live_parks(lifecycle, row["thread_id"])
    except Exception:  # noqa: BLE001
        parks, next_nodes = [], []
    if parks or status == "interrupted":
        B.mark_status(db, build_id, "parked", error=None, terminal=False)
        return "parked"
    if status == "error":
        B.mark_status(db, build_id, "error", error=(lifecycle.error_message or "graph error")[:2000])
        return "error"
    if status == "paused":
        B.mark_status(db, build_id, "parked", error="paused by the operator", terminal=False)
        return "parked"
    try:
        snap = await lifecycle.graph.aget_state(_config(row["thread_id"]))
        values = (snap.values if snap else {}) or {}
    except Exception:  # noqa: BLE001
        values = {}
    entries = [b for b in values.get("completed_blocks") or [] if b.get("name") == row["module"]]
    if not entries:
        if next_nodes:
            B.mark_status(db, build_id, "parked", error="stopped at a node boundary; resume to continue", terminal=False)
            return "parked"
        B.mark_status(db, build_id, "error", error="the graph ended without a block result")
        return "error"
    last = entries[-1]
    if last.get("success"):
        # block_done committed the pass; the row is completed unless the
        # commit failed (then block_done reported a persistence failure)
        row = B.get_build(db, build_id)
        if row and row["status"] == "completed":
            return "completed"
        B.mark_status(db, build_id, "failed_persistence",
                      error="the block result reports success but no completed build record was committed")
        return "failed_persistence"
    if last.get("persistence_failed"):
        B.mark_status(db, build_id, "failed_persistence", error=str(last.get("error") or "")[:2000])
        return "failed_persistence"
    if last.get("aborted") or last.get("skipped"):
        B.mark_status(db, build_id, "aborted", error=str(last.get("error") or ("skipped" if last.get("skipped") else "aborted"))[:2000])
        return "aborted"
    B.mark_status(db, build_id, "failed", error=str(last.get("error") or "block failed")[:2000])
    return "failed"


async def watch_build(db, root, lifecycle, build_id: str, *, pipeline=None) -> str:
    """Await the lifecycle's task, classify the outcome, merge a completed
    build into the pipeline checkpoint (when a pipeline is given) and emit
    the terminal/parked graph event. Never raises."""
    try:
        task = getattr(lifecycle, "task", None)
        await wait_task(lifecycle)
        _release_build_lease(db, lifecycle, task=task)
        outcome = await finalize_after_task(db, root, lifecycle, build_id)
        if outcome == "completed" and pipeline is not None:
            await merge_into_pipeline(pipeline, lifecycle, db, build_id)
        try:
            from orchestrator.langgraph.event_stream import write_graph_event
            write_graph_event(str(root), "Build", "build_terminal" if outcome in B.BUILD_TERMINAL else "build_parked",
                              {"build_id": build_id, "status": outcome})
        except Exception:  # noqa: BLE001
            pass
        return outcome
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("build watch failed for %s", build_id, exc_info=True)
        return "unknown"


async def recover_builds(db, root, lifecycle) -> list[dict]:
    """After a process restart: every build row the build graph dispatched
    that is not terminal is re-read from its persistent checkpoint --
    ``parked`` when an interrupt waits or the graph stopped at a node
    boundary (resumable), ``completed``/``failed``/... when the graph
    finished while the old process was gone, and ``error`` when the row was
    dispatched but no checkpoint was ever written (the process died before
    the graph started): a truthful start failure that keeps its history and
    lets the module be built again, never an in-flight build that blocks it
    forever."""
    out = []
    async with _op_lock():
        # leases this process (or a dead one on this host) left behind
        for name in GRAPH_LEASES.values():
            row = db.lease(name) if hasattr(db, "lease") else None
            if row and _live_lease(db, name) is None:
                db.steal_lease(name, "holder gone; released at recovery")
        for row in B.active_builds(db):
            if row.get("graph") != BUILD_GRAPH:
                continue
            if not row.get("thread_id"):
                B.mark_status(db, row["id"], "error", error="dispatched without a graph thread")
                out.append({"build_id": row["id"], "status": "error", "checkpoint": None})
                continue
            try:
                status = await lifecycle.select_thread(row["thread_id"])
            except Exception as exc:  # noqa: BLE001
                out.append({"build_id": row["id"], "status": row["status"], "error": str(exc)[:200]})
                continue
            if status == "idle":
                B.mark_status(db, row["id"], "error",
                              error="dispatched but never checkpointed: the process ended before the graph started; "
                                    "start a new build")
                new = "error"
            elif status in ("interrupted", "done", "paused"):
                new = await finalize_after_task(db, root, lifecycle, row["id"])
            else:
                new = row["status"]
            out.append({"build_id": row["id"], "thread_id": row["thread_id"], "checkpoint": status, "status": new})
    return out


def build_state(db, root, build_id: str, *, lifecycle=None, parks: list[dict] | None = None,
                next_nodes: list[str] | None = None) -> dict:
    """The durable record first, the live lifecycle second: ``status`` is
    the ``builds`` row's; ``lifecycle`` reports the in-process task when it
    is driving this build's thread; ``current`` re-checks a completed
    build's inputs and produced implementation against the project now."""
    row = B.get_build(db, build_id)
    if row is None:
        raise BuildRefusal("UNKNOWN_BUILD", f"no build {build_id}", http_status=404)
    out = {"build": row, "status": row["status"], "module": row["module"],
           "current": B.current_build_status(db, root, row["module"]) if row["status"] == "completed" else None,
           "interrupts": parks or [], "next_nodes": next_nodes or []}
    if row["status"] in B.BUILD_ACTIVE:
        refusal = stale_refusal(db, root, row)
        out["resumable"] = refusal is None
        out["stale"] = [b["text"] for b in refusal.blockers] if refusal else []
        if refusal and parks and all((p.get("payload") or {}).get("type") == "acceptance_changed" for p in parks) \
                and stale_refusal(db, root, row, allow_axes={"acceptance"}) is None:
            out["resumable"] = True
            out["resume_actions"] = ["restore", "abort"]
    if lifecycle is not None:
        out["lifecycle"] = {"name": lifecycle.name, "thread_id": lifecycle.thread_id, "status": lifecycle.status,
                            "driving_this_build": lifecycle.thread_id == row["thread_id"],
                            "task_running": _running(lifecycle),
                            "error_message": lifecycle.error_message or None}
    if row["status"] == "completed":
        out["evidence"] = B.evidence_for_build(root, build_id)
    out["targets"] = targets_summary(row.get("inputs") or {})
    out["candidates"] = candidate_summaries(db, build_id)
    return out


def candidate_summaries(db, build_id: str) -> list[dict]:
    """Every evaluated candidate of a build, compact: outcome, measured
    targets with their gaps."""
    out = []
    for c in B.candidates_for(db, build_id):
        ev = c.get("evaluation") or {}
        out.append({"id": c["id"], "attempt": c["attempt"], "outcome": c["outcome"],
                    "ts": c["ts"],
                    "missed": ev.get("missed") or [], "unmeasured": ev.get("unmeasured") or [],
                    "targets": {t["id"]: {"status": t["status"], "value": t["value"], "unit": t["unit"],
                                          "required": t["required"], "gap": t.get("gap"), "reasons": t.get("reasons")}
                                for t in ev.get("targets") or []},
                    "functional": {f["id"]: f["status"] for f in ev.get("functional") or []}})
    return out


async def state_with_parks(db, root, lifecycle, build_id: str) -> dict:
    """:func:`build_state` with the live parks of an active build read from
    its checkpoint (not while the lifecycle's task is driving that thread)."""
    row = B.get_build(db, build_id)
    if row is None:
        raise BuildRefusal("UNKNOWN_BUILD", f"no build {build_id}", http_status=404)
    parks, next_nodes = [], []
    driving = _running(lifecycle) and lifecycle.thread_id == row["thread_id"]
    if row["status"] in B.BUILD_ACTIVE and row.get("graph") == BUILD_GRAPH and not driving:
        try:
            parks, next_nodes = await live_parks(lifecycle, row["thread_id"])
        except Exception:  # noqa: BLE001
            parks, next_nodes = [], []
    return build_state(db, root, build_id, lifecycle=lifecycle, parks=parks, next_nodes=next_nodes)


def resume_value(action: str, *, feedback: str = "", rtl_fix_description: str = "", rationale: str = "",
                 actor: str = "cli") -> dict:
    return {"action": action, "feedback": feedback or "", "rtl_fix_description": rtl_fix_description or "",
            "rationale": rationale or "", "actor": actor or "cli", "resumed_at": time.time()}


def checkpoint_db_path(root) -> str:
    return os.path.join(str(root), ".coresmith", BUILD_CHECKPOINT_DB)


async def merge_into_pipeline(pipeline, build_lifecycle, db, build_id: str) -> bool:
    """Append a completed build's block result to the pipeline checkpoint's
    ``completed_blocks`` (the channel's reducer appends; the gates dedup by
    name keeping the last entry), so ``pipeline_complete`` sees the recorded
    pass -- and verifies its receipt. A no-op without a pipeline checkpoint.
    Best-effort: False on failure, never raises."""
    row = B.get_build(db, build_id)
    if row is None or row["status"] != "completed":
        return False
    try:
        snap = await build_lifecycle.graph.aget_state(_config(row["thread_id"]))
        entries = [b for b in ((snap.values if snap else {}) or {}).get("completed_blocks") or []
                   if b.get("name") == row["module"] and b.get("build_id") == build_id]
        if not entries:
            return False
        await pipeline.ensure_graph()
        pconf = _config(pipeline.thread_id)
        psnap = await pipeline.graph.aget_state(pconf)
        if not psnap or not psnap.values:
            return False
        await pipeline.graph.aupdate_state(pconf, {"completed_blocks": [entries[-1]]}, as_node="process_block")
        return True
    except Exception:  # noqa: BLE001
        return False


async def wait_task(lifecycle) -> None:
    """Await the lifecycle's current task without raising."""
    task = lifecycle.task
    if task is None:
        return
    try:
        await asyncio.shield(task)
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - the task records its own outcome
        pass
