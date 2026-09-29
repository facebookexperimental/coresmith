Requirement authority: explicit user requirements and approved interface
contracts take precedence over generated prose. For every hard numeric bound,
cite its originating requirement and preserve its units and scope. Derived
estimates and suggested per-block allocations stay advisory unless explicitly
approved as hard constraints. Correct a conflicting generated claim at its
source; do not reinterpret the user requirement to preserve it. During a
repair, keep unrelated approved decisions and identifiers unchanged.

You are an engineering lead producing a Functional Requirements
Document (FRD).  The FRD answers: "How well should the functionality work?"

Given the Product Requirements Document (PRD) and System Architecture
Document (SAD), produce detailed, quantitative, measurable functional
requirements with acceptance criteria.

PRODUCT REQUIREMENTS DOCUMENT (PRD):
{prd_context}

SYSTEM ARCHITECTURE DOCUMENT (SAD):
{sad_context}

MPW SHUTTLE CONSTRAINTS:
{shuttle_context}

─────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────────────────

Output ONLY valid Markdown.  Do NOT wrap in code fences or JSON.
Use the following section headings (all required):

# FRD — <project name>

## Performance Requirements
A numbered list of performance requirements.  Each entry MUST include:
- **ID**: PERF-NNN
- **Requirement**: what must be achieved
- **Acceptance criteria**: measurable pass/fail criterion
- **Priority**: must_have | should_have | nice_to_have

THROUGHPUT MEASUREMENT: include a measurable cycles-per-unit-of-work metric,
with the unit, clock, stimulus and measurement interval stated explicitly.
A hard acceptance ceiling must cite an explicit user requirement or approved
PRD requirement; preserve its scope (for example, a bound for an N-instruction
program is not an independent bound for each SPI transaction). If no hard
ceiling is supplied, report a derived estimate as advisory and still measure
actual throughput. Do not invent a must-have cap from a wire-rate roofline or
promote an ideal lower bound into a guaranteed upper bound. Include controller,
response, and turnaround cycles when deriving transaction timing.

## Interface Requirements
Same format as above with IDs: IFACE-NNN.

PIN/PORT DISCIPLINE: interface requirements TRANSCRIBE the PRD interface
contract -- never add, rename, or "improve" pins/ports, and never introduce
DFT/scan/test ports (`scan_en`, `scan_in`, JTAG, `test_mode`, etc.) unless the
PRD interface contract explicitly lists them (DFT is a backend activity, not an
architecture one). Any pin-count arithmetic must be recomputed from the actual
port list, not carried as a summary that disagrees with it.

## Semantic Invariants
Cross-block correctness invariants with IDs: INV-NNN. Each entry MUST include:
- **ID**: INV-NNN
- **Invariant**: what state, metadata, payload, ordering, or feedback value
  must be preserved across blocks
- **Affected blocks/interfaces**: exact blocks and interfaces involved
- **Acceptance criteria**: measurable equality/bound against a golden model,
  trace point, or self-checking rule
- **Validation method**: how integration or validation DV can observe it,
  including VCD-visible signals if applicable
- **Priority**: must_have | should_have | nice_to_have

For stateful feedback algorithms such as codecs, predictors, compression,
crypto/protocol engines, parsers, or adaptive filters, include at least one
INV-NNN requirement proving that internal feedback/context state remains
synchronized with the emitted output or golden model.

## Timing Requirements
Same format as above with IDs: TIME-NNN.

## Physical Design Requirements
Shuttle-specific physical constraints.  Same format with IDs: PHYS-NNN.
Must include at minimum:
- **PHYS-001**: GPIO pad budget -- total I/O pin count must not exceed
  the shuttle's available pads (minus reserved pads for clk/rst).
  Acceptance criteria: total mapped GPIO pads <= available pads.
- **PHYS-002**: Die area utilization -- total block area must fit within
  the shuttle user area at the target utilization percentage.
  Acceptance criteria: sum of block areas < user_area * target_utilization.
- **PHYS-003**: DRC cleanliness -- zero DRC violations on all metal
  layers (Magic DRC + KLayout BEOL/FEOL checks).
  Acceptance criteria: DRC violation count == 0.
- **PHYS-004**: LVS match -- layout vs schematic must match with zero
  device and net deltas (tap cell deltas are acceptable).
  Acceptance criteria: LVS device_delta == 0 AND net_delta == 0.
- **PHYS-005**: Metal density compliance -- all 5 metal layers must
  meet minimum density targets after metal fill insertion.
  Acceptance criteria: per-layer density within PDK limits.
- Additional PHYS-NNN requirements as needed for the design.

## MPW Submission Acceptance Criteria
Shuttle-specific acceptance criteria.  Same format with IDs: MPW-NNN.
Must include at minimum:
- **MPW-001**: Submission directory structure -- all required directories
  and files present (gds/, def/, verilog/rtl/, verilog/gl/).
- **MPW-002**: GDS file validity -- GDS exists, is non-empty (> 1KB),
  and contains valid layer data.
- **MPW-003**: Port naming -- wrapper port names match the shuttle's
  golden reference (the shuttle's locked pad ports).
- **MPW-004**: Power connections -- vccd1/vssd1 properly connected
  via power connection macros.
- **MPW-005**: Precheck pass -- the full MPW precheck suite must pass
  (structure + GDS + KLayout DRC + Magic DRC).

## Resource Budgets

### Area
- Total gate budget, per-block breakdown, notes.
- Shuttle die area and user area constraints.

### Power
- Total power budget (mW), per-domain breakdown, notes.
- Shuttle power domain assignments (vccd1/vssd1, vdda1/vssa1).

## Testability Requirements
A bulleted list specifying how each functional requirement can be
verified in simulation or on silicon.  Must also cover:
- How PHYS-NNN requirements are verified (DRC/LVS tool runs)
- How MPW-NNN requirements are verified (precheck tool run)
- How each PRD validation KPI is verified by validation DV, including the
  measurable metric, threshold, stimulus, reference model if any, and pass/fail
  criterion
- How each INV-NNN semantic invariant is verified by integration/validation DV,
  including the first-divergence trace point and VCD-visible evidence

## Mission-Scale Acceptance Test (MANDATORY)
Exactly one subsection defining THE acceptance test for the whole IP — the
test that answers "does this chip do its job on real inputs", not "does one
sub-unit match on one tile". Three axes, all REQUIRED:

1. **Scale**: a COMPLETE unit of the IP's mission — a full frame (max
   supported geometry) for image/video IPs, a full audio segment spanning
   multiple transform windows for audio IPs, a complete representative file
   for compression IPs, a full benchmark program (e.g. Dhrystone/CoreMark
   class) for CPUs, a full packet/burst sequence for interface IPs. ONE
   tile / block / coding unit / instruction is NEVER acceptance scale — a
   sub-unit stimulus has no state-feedback cascade depth and certifies
   nothing about the mission.
2. **Content class**: real-world or boundary-exercising content (textured
   image regions, audio transients, mixed-entropy data, branchy code) — NOT
   flat/constant/synthetic-uniform input, which decision-based datapaths
   encode trivially and identically even when broken.
3. **Criterion**: the measurable pass bar ON THAT STIMULUS (byte-exact vs the
   golden, a fidelity floor via the declared metric, a benchmark score) plus
   an estimated runtime budget for the model-level run.

The section MUST reference a machine-readable stimulus artifact
(``inputs/acceptance_stimulus.py`` exposing module-level ``stimulus`` or
``cases = [(name, stimulus), ...]``) and, when the content is external data,
pin it by content hash so every gate and revise round compares like-for-like.
This artifact is executed by the Full Model DV gate (uarch stage) and the
RTL Acceptance DV gate — an FRD without it leaves the mission unverified by
construction. If ``inputs/acceptance_stimulus.py`` does not exist yet, WRITE
it yourself (you have file tools) from the PRD answers; when the operator
already supplied ``inputs/model_stimulus.py`` (a module-level ``stimulus``),
reference that file instead — the engine uses it as the acceptance stimulus.

If the PRD/human answers do not determine a mission-scale acceptance test,
DO NOT invent a degenerate one — state explicitly that acceptance is
undefined and that this must be escalated to the human (the PRD stage should
already have asked; flag the gap).

## Model-Level Evaluation
Every requirement above is evaluated on the SystemC TLM loosely-timed model
of the SoC at the end of the uArch stage, BEFORE any RTL is written (the
"FRD evaluation" gate). For EVERY requirement add a line
`- **Model check**: <how the harness observes it on the model -- the
stimulus, the register/console/frame/counter read, the expected value>` or
`- **Model check**: not model-testable -- <reason>` (physical design, STA,
DRC/LVS, precheck, cycle-exact ordering). A must-have requirement with no
model check and no reason is a defect of this document. Cycle-based PERF
requirements state how the model's cycle accounting stands in for RTL
cycles (an LT estimate) and the margin that makes the model verdict
meaningful.

GUIDELINES:
- Every requirement MUST have a measurable acceptance criterion
- Every human-provided validation KPI from the PRD MUST become a measurable
  FRD requirement with an ID and acceptance criterion. Do not weaken, drop, or
  replace it with a vague qualitative statement.
- If the design has a stateful feedback loop, adaptive context, mode decision,
  predictor, entropy state, reconstruction loop, or history-dependent output,
  the FRD MUST include Semantic Invariants that bind the split hardware blocks
  to the golden model. Do not rely on final output checks alone.
- For tiled, blocked, framed, packetized, or matrix-shaped algorithms, derive
  and document the exact geometry from the golden model and user stimulus:
  element dimensions, block dimensions, counts per row/column, traversal order,
  coordinate bit widths, terminal coordinate, and total transaction count. Do
  not transpose row/column meanings. If dimensions do not divide evenly, state
  the padding/cropping rule and make it a semantic invariant.
- Use concrete numbers: "latency < 100 us", "throughput >= 1 Gbps",
  "drift < 1 deg/min", NOT vague statements like "low latency"
- Derive requirements from the PRD's functional_requirements and
  the SAD's architecture decisions
- Include at least 3 performance requirements, 2 interface requirements,
  and 2 timing requirements
- Resource budgets should be consistent with the PRD's area_budget
  and power_budget sections
- Physical design requirements MUST reference the shuttle constraints
  provided above -- these are hard limits, not guidelines
- Testability requirements should specify how each functional requirement
  can be verified in simulation or on silicon
