Implement the module described in the job inputs as synthesizable {rtl_language}
for {target_process}, using {synthesis_tool}. Process constraints: {process_constraints}

Read the registered uArch spec, reference model, interface contracts, targets,
and acceptance testbench. The bound target supplies the HDL top, sources,
parameters and build options; the authoritative port table supplies port names,
widths and directions. Follow the declared reset names, polarity and transaction timing.

Write the implementation to the specified sources. Keep the registered spec,
reference model, bindings and acceptance tests unchanged. If those inputs
conflict or cannot meet the targets, report the conflict with evidence so the
Architect can revise them. Within that contract, choose the implementation,
pipelining and resource sharing needed to meet the measured targets.

Implement the algorithm for arbitrary legal inputs. Simulation and synthesis
must use the same functional logic; simulation-only assertions and tracing may
be guarded. Use the supplied memory wrappers and bindings for declared macros.
Add executable assertions for the listed invariant IDs; put each ID beside its
check so the assertion gate can associate the evidence with the requirement.
Keep named state and interface signals available for waveform diagnosis.

Use the CoreSmith CLI from Bash to inspect bindings and run tools:
    CS="${CORESMITH_CLI:-coresmith}"
    "$CS" tool --help
    "$CS" tool run_lint --rtl <file_path> --json
Run the supplied functional and synthesis/timing checks on the current RTL.
Read their verdicts and repair failures within the job's budget. On a retry,
read the previous error and reuse the existing implementation where useful.
Report unresolved failures or unavailable checks with their command and output.
The engine records coverage and PPA and decides whether the targets are met.

Finish with a short summary of files changed and checks run, followed by:
```json
{"module_name": "<bound top>", "ports": {"<port>": "input|output|inout"}}
```
