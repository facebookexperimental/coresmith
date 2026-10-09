# Targets and build outcomes

The Architect allocates module targets in the functional requirements document (FRD) before a build. Each target binds a quantity, unit, bound, owner, and measurement method; `build module` consumes that allocation unchanged.

| Quantity | Measurement | Scope |
|---|---|---|
| Throughput or latency | Bound cocotb test calls `measure.record` | Declared workload and clock |
| Area | Yosys report plus bound macro area | Mapped netlist; explicit standard-cell subtotal supported |
| Power | OpenSTA `report_power` | Vectorless estimate; library, clock, activity and duty recorded |

## Bind targets

Use `frd add` to declare the bound and `frd verifier` to bind its measurement. `build targets <module>` reports the allocation and any blockers before the build starts.

## Build outcomes

Targets are judged after synthesis, from this attempt's tool receipts. Required FRD bounds are checked before publication.

| Result | Build |
|---|---|
| DV, coverage, synthesis, timing and every required target pass | Completes and publishes |
| A required target is measured and missed | Returns the gap to the RTL worker as a repair attempt |
| A required target cannot be measured | Parks `target_unmeasured`; `retry` after fixing the tool environment |
| Repair attempts are exhausted | Fails |

Advisory targets are measured where possible and reported; they never gate the build. Changing a target's bound or measurement is a new input and requires a new build.

Code: `orchestrator/state_store/module_targets.py`, `orchestrator/langgraph/target_closure.py`.
