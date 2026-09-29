# Migrating a project to the declarative task contract

This release removes several architecture stages, makes acceptance the task's own
responsibility, and replaces a set of heuristics with explicit declarations. A project
created before it keeps working only if it declares the things the engine used to guess.
Everything below is a task-side change; no engine flags are required.

## 1. Declare the chip top

The engine no longer infers the top module from the RTL it finds.

```yaml
# inputs/task.yaml
top: user_project_wrapper     # the module the task is graded on
```

`CORESMITH_TOP_MODULE` overrides the file. If a declared top is not a module any block
RTL defines, the integration check parks instead of inventing a wrapper. With no
declaration at all, a single-block design still gets a generated passthrough wrapper
named after the design, as before.

## 2. Declare the chassis, or say there is none

```yaml
chassis: caravel   # or: openframe | accel (a Caravel alias) | none
```

An absent key and `none` both mean no chassis: no locked boundary ports, no pin-boundary
expectations, and no chassis assembly. An unknown name is an error rather than a silent
fallback. `CORESMITH_CHASSIS` is consulted only when the file does not declare one. The
older `CORESMITH_DETERMINISTIC_CARAVEL_TOP=1` flag is not a chassis declaration: a task
that relied on it must now declare `chassis: caravel`.

## 3. Acceptance: ship a task adapter

The engine's built-in stream harness is no longer the acceptance authority. A task ships
`inputs/task_adapter.py`; its verdict outranks the engine's internal validation
requirements. The contract lives in `orchestrator/harness/task_adapter.py`, and a minimal
runnable example is in [examples/task_adapters](../examples/task_adapters).

The adapter runs in its own interpreter, inside a bubblewrap boundary that leaves only
its work directory writable. Install `bubblewrap`, or set
`CORESMITH_ADAPTER_SANDBOX=none` to opt out deliberately; a missing sandbox with no
opt-out fails the gate rather than running unconfined. If the host's `PATH` puts an EDA
toolchain first and that bundle ships its own `bwrap`, point `CORESMITH_BWRAP` at the
system one.

Receipts are written under `.coresmith/acceptance/<candidate-sha>/attempt-NNNNNN/`, so a
consumer that used to read a single fixed path must walk attempts.

## 4. Native stream acceptance requires an explicit mapping

Tasks without an adapter that use the built-in stream path must declare how their
payload is packed; the engine no longer infers geometry from array shape or masks
values to bytes. On the stimulus module:

```python
AXIS_MAPPING = {
    "payload": None,          # attribute holding the payload, or None for the case value
    "input_width": 32,        # DUT input width in bits, a multiple of 8, 8..64
    "output_width": 32,
    "packing": "words",       # "words" | "bytes"
    "byte_order": "little",
    "sidebands": {},          # exact scalar field -> signal mapping
}
```

An undeclared or unsupported mapping reports an incomplete oracle and parks. A task
adapter is the supported route for anything this cannot express.

## 5. Optional: certify maximum geometry

Maximum-geometry coverage is opt-in and, when absent, non-blocking and reported as not
certified.

```yaml
max_geometry_cases:
  depth_limit: {queue_depth: 64}
```

Only a successfully executed case of that name can cover a dimension. A malformed
declaration is an error. Do not edit `inputs/` to satisfy a gate: inputs are owner
material and the chip lead is instructed never to modify them.

## 6. Candidate receipts and the integrity baseline

Integration adopts one validated candidate manifest at
`.coresmith/candidate.json`: the declared top, the exact source list, the include and
`$readmem` closure, and the defines and parameters in force. Every consumer reads it, so
a project copied to a new path, or whose RTL changed outside the pipeline, must be
re-adopted by re-running the integration check.

The oracle integrity baseline now lives outside the project, by default under
`~/.coresmith/trust/<project-id>/`, and is captured when a run is started through an
owner entry point. Override the location with `CORESMITH_TRUST_DIR`. A missing baseline
fails closed; resuming an old project whose baseline lived inside the project requires
starting a fresh trusted run.

## 7. Removed stages and artifacts

The uArch exploration stage and the Memory Map, Clock Tree, Register Spec and Complexity
Review stages are gone, along with the WaveKit audit, the regex lint pass, the staleness
ledger and the gate-retry bypasses. Their artifacts are no longer produced, and a
project that reads `memory_map.json`, `clock_tree.json` or `register_spec.json` should
drop those reads. Project state is one SQLite database at `.coresmith/project.sqlite`;
the JSON files beside it are regenerated views. Editing a view on disk is supported: the
next write imports the edit before regenerating.

## 8. What is stricter now

- The coverage floor and golden-required signoff are hard gates again.
- An acceptance failure, an exhausted feedback budget, or a required oracle that cannot
  run parks for a decision instead of being skipped.
- A block whose ports do not match the frozen interface contract does not reach
  integration.
- A required block must appear in the elaborated hierarchy, checked with Yosys rather
  than by matching source text.
- Reported timing says `unknown` rather than showing a leaf estimate as top-level Fmax,
  and a gate-level replay that compares only part of the reference is `bounded`, not a
  pass.
