# Socmate Pipeline

Socmate is the ASIC generation pipeline inside this repository. The codebase is
currently named `coresmith`, and new service endpoints are being added under
`coresmithd`, but the core pipeline described here is the same disk-first
LangGraph flow.

The pipeline turns requirements or a block registry into ASIC collateral:
architecture documents, per-block microarchitecture specs, RTL, cocotb tests,
simulation results, synthesized netlists, integrated top-level RTL, physical
design artifacts, and optional OpenFrame tapeout output.

## Documentation Order

This GitBook starts with the pipeline itself:

- [Pipeline Overview](pipeline-overview.md)
- [Architecture Phase](architecture-phase.md)
- [Frontend RTL Pipeline](frontend-rtl-pipeline.md)
- [Backend And Tapeout](backend-and-tapeout.md)
- [State, Artifacts, And Observability](state-artifacts-observability.md)

The final section covers how to drive the same pipeline through either
`coresmithd` or MCP:

- [Usage](usage.md)

## Mental Model

One project root owns one run state:

```text
project root
  .coresmith/             checkpoint DBs, events, traces, transient state
  arch/                   generated architecture and uArch documents
  rtl/                    generated block RTL and integration RTL
  tb/cocotb/              generated block and top-level testbenches
  sim_build/              simulation build output
  syn/output/             synthesis and backend output
  openframe_submission/   optional tapeout collateral
```

The graphs carry routing metadata in LangGraph state. Long-form content lives
on disk so outer agents, daemon clients, MCP clients, and humans can inspect and
edit the same files before resuming a paused run.

