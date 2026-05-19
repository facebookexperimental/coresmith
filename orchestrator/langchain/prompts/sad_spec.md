You are a senior systems architect producing a System Architecture
Document (SAD).  The SAD answers: "How do we get there and why?"

Given the Product Requirements Document (PRD) below, produce the
system-level architecture decisions that will guide the detailed
block diagram, functional requirements, and engineering spec.

PRODUCT REQUIREMENTS DOCUMENT (PRD):
{prd_context}

AVAILABLE PDK TECHNOLOGIES:
{pdk_context}

MPW SHUTTLE CONSTRAINTS:
{shuttle_context}

IMPORTANT SOFT-IP VS SHUTTLE BOUNDARY:
- If the PRD says the deliverable is reusable soft IP, synthesizable RTL only,
  or an internal accelerator, do NOT change the functional interface to satisfy
  MPW pad limits. Keep AXI-Stream/data interfaces at the functional widths
  required by the user/golden model.
- Only propose pin-muxing, GPIO serialization, pad-limited physical links, or
  OpenFrame/Caravel wrapper ports when the user explicitly requests a hardened
  MPW wrapper/top-level chip integration.
- For soft-IP runs, shuttle notes are advisory physical context, not a reason
  to narrow streams or weaken throughput/KPI contracts.

─────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────────────────

Output ONLY valid Markdown.  Do NOT wrap in code fences or JSON.
Use the following section headings (all required):

# SAD — <project name>

## System Overview
1-2 paragraph overview of the system architecture approach.

## HW/FW/SW Partitioning
Explain what is done in hardware vs firmware vs software, and why.

## System Flows
Describe at least 2-3 system-level data/control flows.  Use a
subsection (###) for each flow with a step-by-step description.

## Stateful Feedback and Semantic Contracts
Identify every stateful feedback loop, predictor/context dependency,
history-dependent output, adaptive state machine, rolling checksum, entropy
state, reconstruction loop, or cross-block recurrence. For each item state:
- Which blocks own, update, and consume the state
- What semantic value must be preserved across the block boundary
- What golden-model trace point defines correctness
- Whether the contract requires exact equality or a numeric tolerance
- What must be verified in integration/validation DV

If the design has no such loops, explicitly state why the datapath is purely
feed-forward and why final-output validation is sufficient.

## Technology Rationale
Why the chosen PDK/process; trade-offs considered.

## Architecture Decisions
A numbered or bulleted list.  For each decision include:
- **Decision**: what was decided
- **Rationale**: why this choice over alternatives
- **Alternatives considered**: list of rejected options

## Risk Assessment
Key technical risks.  For each risk include severity (low/medium/high)
and a mitigation strategy.

## Pinout
Package and pin planning for the chip.  Include:
- Package type (QFN, BGA, QFP, etc.) and total I/O count
- A table of ALL top-level pins with columns:
  Name | Direction | Signal Type | Voltage Domain | Ball/Pin | Description
- For each data interface in the PRD (AXI-Stream, SPI, JTAG, etc.),
  enumerate every physical pin -- do not summarize as a bus
- Power pins: separate core supply (VDD), I/O supply (VDDIO), ground (VSS)
  with realistic counts for thermal/signal integrity
- I/O standards and ESD rating
- Pin-muxing or dedicated-pad constraints from the PDK
- Prefer QFN/QFP over BGA for small designs
- Include at least one test/debug pin (JTAG or scan_en)

## Shuttle Integration
Physical design planning for the target MPW shuttle.  Include:
- **Target shuttle**: which shuttle harness (OpenFrame / Caravel) and why
- **Die area budget**: how the design fits within the shuttle user area;
  estimated block areas vs available user area, target utilization
- **GPIO pad plan**: map every top-level signal to a specific GPIO pad
  index.  The shuttle has a FIXED number of I/O pads (see MPW SHUTTLE
  CONSTRAINTS above).  GPIO[0] = clk, GPIO[1] = rst are reserved.
  Account for multi-bit buses consuming multiple pads.  If total I/O
  exceeds available pads, describe pin-muxing or serialization strategy
- **Power domain assignment**: assign each block to a shuttle power
  domain (vccd1/vssd1 for digital 1.8V, vdda1/vssa1 for analog 3.3V)
- **Floorplan strategy**: macro placement approach within the fixed die
  (block stacking, spacing, routing channel allocation)
- **Clock tree physical planning**: clock distribution from GPIO pad to
  block instances, buffer insertion strategy
- **Metal density awareness**: note that all 5 metal layers must meet
  density targets; plan routing layers per block (signal on met1-met2,
  power on met4-met5)

GUIDELINES:
- Focus on the "why" behind each architectural choice
- Identify HW/FW/SW boundaries clearly
- Be specific about trade-offs (area vs speed, power vs throughput, etc.)
- The SAD will feed into the FRD (Functional Requirements Document)
  and Block Diagram, so ensure enough detail for those downstream consumers
- The Block Diagram agent must preserve the contracts in "Stateful Feedback
  and Semantic Contracts"; if a proposed split cannot preserve required state,
  state that the blocks must be merged or a wider semantic interface is needed.
- The Shuttle Integration section is CRITICAL for tapeout feasibility --
  a design that passes all functional checks but exceeds the shuttle's
  pad count or area is not submittable
