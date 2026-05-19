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

## Interface Requirements
Same format as above with IDs: IFACE-NNN.

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
  golden reference (io_in, io_out, io_oeb for OpenFrame).
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
