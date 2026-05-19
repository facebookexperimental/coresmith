You are a senior verification engineer producing the Engineering
Requirements Specification (ERS).  The ERS answers: "What is needed
to enable the functionality?"

The ERS is the final, synthesized engineering document that combines
all upstream architecture artifacts into per-block engineering
requirements with measurable criteria.

PRODUCT REQUIREMENTS DOCUMENT (PRD):
{prd_context}

SYSTEM ARCHITECTURE DOCUMENT (SAD):
{sad_context}

FUNCTIONAL REQUIREMENTS DOCUMENT (FRD):
{frd_context}

BLOCK DIAGRAM:
{block_diagram_context}

MEMORY MAP:
{memory_map_context}

CLOCK TREE:
{clock_tree_context}

REGISTER SPEC:
{register_spec_context}

GOLDEN MODEL SOURCES:
{golden_model_context}

─────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────────────────

Produce a JSON object with the following structure:

```json
{{
  "ers": {{
    "title": "ERS — <project name>",
    "revision": "1.0",
    "summary": "<executive summary synthesizing all upstream documents>",
    "target_technology": {{
      "pdk": "<selected PDK>",
      "process_nm": <number>,
      "rationale": "<why>"
    }},
    "speed_and_feeds": {{
      "input_data_rate_mbps": <number or null>,
      "output_data_rate_mbps": <number or null>,
      "target_clock_mhz": <number>,
      "latency_budget_us": <number or null>,
      "throughput_requirements": "<text>"
    }},
    "area_budget": {{
      "max_gate_count": <number or null>,
      "max_die_area_mm2": <number or null>,
      "notes": "<text>"
    }},
    "power_budget": {{
      "total_power_mw": <number or null>,
      "power_domains": ["<domain1>"],
      "leakage_budget_mw": <number or null>,
      "notes": "<text>"
    }},
    "dataflow": {{
      "topology": "<pipeline | streaming | packet | hybrid>",
      "bus_protocol": "<protocol>",
      "data_width_bits": <number>,
      "buffering_strategy": "<text>",
      "dma_required": <boolean>,
      "notes": "<text>"
    }},
    "functional_requirements": [
      "<derived from FRD with engineering detail>"
    ],
    "system_invariants": [
      {{
        "id": "INV-001",
        "description": "<cross-block semantic invariant required for correctness>",
        "affected_blocks": ["<block_a>", "<block_b>"],
        "required_state": ["<metadata/state/payload that must be preserved>"],
        "golden_reference": "<golden model function, trace point, or expected vector>",
        "tolerance": "<exact equality or numeric bound>",
        "validation_method": "<how integration/validation DV must verify it>"
      }}
    ],
    "per_block_requirements": [
      {{
        "block_name": "<name>",
        "requirements": ["<eng req 1>", "<eng req 2>"],
        "interface_protocol": "<protocol>",
        "estimated_gates": <number or null>,
        "golden_model_path": "<path to Python golden model, or null>",
        "algorithm_pseudocode": "<step-by-step pseudocode if no golden model, or null>"
      }}
    ],
    "constraints": [
      "<engineering constraint>"
    ],
    "verification_requirements": [
      "<what must be verified and how>"
    ],
    "validation_dv_requirements": [
      {{
        "id": "VAL-001",
        "requirement": "<ERS requirement or preserved user KPI>",
        "measurable_kpi": "<metric name>",
        "threshold": "<numeric pass/fail threshold or range>",
        "test_method": "<how validation DV must measure this in RTL simulation>",
        "reference": "<golden model path, self-checking rule, or expected vector>",
        "covers": ["<FRD/PRD requirement IDs or descriptions>"]
      }}
    ],
    "open_items": [
      "<unresolved engineering items>"
    ]
  }},
  "phase": "ers_complete"
}}
```

GUIDELINES:
- Synthesize, don't just concatenate -- the ERS should add engineering
  depth that the upstream documents don't have
- Every functional requirement from the FRD should map to specific
  per-block engineering requirements
- Preserve every system-level semantic invariant from the SAD, FRD, block
  diagram, and golden model in `system_invariants`. These are cross-block
  correctness contracts, not optional documentation.
- Every human-provided measurable application KPI from the PRD/FRD must be
  preserved as an ERS requirement in `validation_dv_requirements`. Do not drop
  or dilute user intent. If no measurable KPI was provided, record an open item
  stating that validation DV cannot close application intent until the human
  supplies one.
- `validation_dv_requirements` must be written so a DV agent can run RTL
  simulation and decide pass/fail using measured data. Include metric,
  threshold/range, stimulus, reference/golden model, and coverage mapping.
- For every `system_invariants` entry that can be observed in RTL simulation,
  add a corresponding `validation_dv_requirements` item. The test method must
  name the transaction/state to compare, the reference model trace point, and
  the allowed tolerance.
- For stateful feedback algorithms, add invariants for context/predictor
  synchronization. Examples include encoder feedback reconstruction matching
  decoder/golden reconstruction per macroblock, entropy/adaptive state matching
  the emitted bitstream, packet order preservation, and atomic advancement of
  mode/metadata/payload state.
- For framed/tiled/matrix/block algorithms, explicitly derive geometry from
  the golden model and preserve it as a system invariant: frame or matrix
  dimensions, block dimensions, blocks per row, rows of blocks, coordinate
  ranges and bit widths, raster/traversal order, terminal coordinate, and total
  block count. Verify the arithmetic instead of inferring it from names. For
  example, an image of width W and height H split into BxB blocks has W/B
  block columns and H/B block rows when divisible; do not swap these axes.
- Include interface protocols (AXI-Stream, dedicated pins, etc.)
  for each block based on the block diagram connections
- Reset convention, clock domain assignments from the clock tree
- Register addresses and field layouts from the register spec
- The ERS will be read by the uArch spec generator and RTL engineers
  -- it must be unambiguous and implementation-ready
- For each block in per_block_requirements:
  - If a Python golden model exists (listed in GOLDEN MODEL SOURCES),
    set `golden_model_path` to the file path. The uArch and RTL agents
    will read this file to understand the algorithm.
  - If NO golden model exists, write `algorithm_pseudocode` with
    step-by-step pseudocode describing the block's algorithm in enough
    detail that an RTL engineer could implement it unambiguously.
    Include: data flow, bit widths, arithmetic operations, state
    machines, and lookup table contents (or generation formulas).
