# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FFT16 Reference Design constants for the coresmith orchestrator test suite.

A tiny, concrete 16-point FFT design that all tests share. Keeps test
data in one place so mock outputs are consistent across test files.

Naming convention:
  FFT16_PRD_*  -- Product Requirements Document (formerly ERS Phase 1/2)
  FFT16_SAD_*  -- System Architecture Document (new)
  FFT16_FRD_*  -- Functional Requirements Document (new)
  FFT16_ERS_*  -- backward-compat aliases for PRD fixtures (remove after refactor)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# Requirements & PRD Answers
# ═══════════════════════════════════════════════════════════════════════════

FFT16_REQUIREMENTS = """\
Design a 16-point FFT processor.
- 16-bit signed fixed-point I/Q input (16-bit real + 16-bit imaginary)
- AXI-Stream input/output interfaces
- Single clock domain, target 50 MHz on sky130
- Radix-2 decimation-in-time butterfly architecture
- On-chip twiddle factor ROM (no external memory)
- Area budget: < 50,000 gates"""

FFT16_PRD_ANSWERS = {
    "target_technology": "sky130 130nm",
    "fft_size": "16-point (fixed, no runtime config)",
    "data_width": "16-bit signed fixed-point",
    "input_data_rate": "50 Msps (one sample per clock)",
    "area_budget": "< 50,000 gates",
    "power_budget": "Not critical for this design",
    "dataflow": "Streaming pipeline, AXI-Stream, no DMA",
}

# Backward-compat alias (remove after ERS->PRD rename in production code)
FFT16_ERS_ANSWERS = FFT16_PRD_ANSWERS

# ═══════════════════════════════════════════════════════════════════════════
# Block Diagram
# ═══════════════════════════════════════════════════════════════════════════

FFT16_BLOCK_DIAGRAM = {
    "blocks": [
        {
            "name": "fft_butterfly",
            "description": "Radix-2 butterfly: add/sub + complex multiply",
            "tier": 1,
            "python_source": "PyDVB/dvb/OFDM.py",
            "rtl_target": "rtl/dvbt/fft_butterfly.v",
            "testbench": "tb/cocotb/test_fft_butterfly.py",
            "interfaces": {"input": {"width": 32}, "output": {"width": 32}},
            "estimated_gates": 2000,
        },
        {
            "name": "twiddle_rom",
            "description": "8-entry sin/cos lookup (16-bit coefficients)",
            "tier": 1,
            "python_source": "PyDVB/dvb/OFDM.py",
            "rtl_target": "rtl/dvbt/twiddle_rom.v",
            "testbench": "tb/cocotb/test_twiddle_rom.py",
            "interfaces": {"addr": {"width": 3}, "data": {"width": 32}},
            "estimated_gates": 800,
        },
        {
            "name": "fft_controller",
            "description": "4-stage pipeline sequencer + bit-reversal",
            "tier": 2,
            "python_source": "PyDVB/dvb/OFDM.py",
            "rtl_target": "rtl/dvbt/fft_controller.v",
            "testbench": "tb/cocotb/test_fft_controller.py",
            "interfaces": {"control": {"width": 8}, "status": {"width": 4}},
            "estimated_gates": 1500,
        },
    ],
    "connections": [
        {"from": "fft_controller", "to": "fft_butterfly", "interface": "control", "data_width": 8},
        {"from": "twiddle_rom", "to": "fft_butterfly", "interface": "axis", "data_width": 32},
        {"from": "fft_butterfly", "to": "fft_controller", "interface": "handshake", "data_width": 1},
    ],
    "questions": [],
}

# ═══════════════════════════════════════════════════════════════════════════
# PRD Specialist Mock Outputs (formerly ERS)
# ═══════════════════════════════════════════════════════════════════════════

FFT16_PRD_QUESTIONS = {
    "questions": [
        {"id": "target_technology", "category": "technology", "question": "Which process technology?",
         "context": "Determines standard-cell library and voltage domain.",
         "options": ["sky130 130nm", "gf180mcu 180nm"], "required": True},
        {"id": "fft_size", "category": "speed_and_feeds", "question": "FFT size?",
         "context": "Determines butterfly count and twiddle ROM depth.",
         "options": ["16-point", "64-point"], "required": True},
        {"id": "data_width", "category": "speed_and_feeds", "question": "Data width?",
         "context": "Affects multiplier area and dynamic range.",
         "options": ["16-bit signed", "32-bit signed"], "required": True},
        {"id": "area_budget", "category": "area", "question": "Max gate count?",
         "context": "Constrains implementation complexity.",
         "options": ["< 50,000 gates", "< 100,000 gates"], "required": True},
        {"id": "power_budget", "category": "power", "question": "Power budget?",
         "context": "Determines clock gating strategy.",
         "options": ["Not critical", "< 50 mW"], "required": False},
        {"id": "dataflow", "category": "dataflow", "question": "Dataflow architecture?",
         "context": "Streaming vs batch affects buffering.",
         "options": ["Streaming pipeline, AXI-Stream", "Batch with SRAM buffer"], "required": True},
    ],
    "phase": "questions",
}

FFT16_PRD_DOCUMENT = {
    "prd": {
        "title": "16-Point FFT Processor PRD",
        "revision": "1.0",
        "summary": "A 16-point radix-2 DIT FFT processor with 16-bit fixed-point I/Q, AXI-Stream, sky130 50 MHz.",
        "target_technology": {"pdk": "sky130", "process_nm": 130, "rationale": "Open-source PDK"},
        "speed_and_feeds": {"target_clock_mhz": 50, "input_data_rate_mbps": 1600,
                            "output_data_rate_mbps": 1600, "latency_budget_us": 0.32,
                            "throughput_requirements": "One sample per clock"},
        "area_budget": {"max_gate_count": 50000, "max_die_area_mm2": 0.5, "notes": "Butterfly dominated"},
        "power_budget": {"total_power_mw": 100, "power_domains": [], "leakage_budget_mw": 10, "notes": "Not critical"},
        "dataflow": {"topology": "pipeline", "bus_protocol": "AXI-Stream", "data_width_bits": 32,
                     "buffering_strategy": "In-place", "dma_required": False, "notes": "No DMA"},
        "functional_requirements": ["16-point radix-2 DIT FFT", "16-bit signed I/Q", "On-chip twiddle ROM"],
        "constraints": ["Single clock domain", "No external memory"],
        "open_items": [],
    },
    "phase": "prd_complete",
}

# Backward-compat aliases (remove after ERS->PRD rename in production code)
FFT16_ERS_QUESTIONS = FFT16_PRD_QUESTIONS
FFT16_ERS_DOCUMENT = {
    "ers": FFT16_PRD_DOCUMENT["prd"],
    "phase": "ers_complete",
}

# ═══════════════════════════════════════════════════════════════════════════
# SAD (System Architecture Document) Mock Output
# ═══════════════════════════════════════════════════════════════════════════

FFT16_SAD_DOCUMENT = {
    "sad": {
        "title": "16-Point FFT Processor SAD",
        "system_overview": (
            "Single-chip radix-2 DIT FFT processor targeting sky130 at 50 MHz. "
            "Composed of a 4-stage butterfly pipeline, twiddle factor ROM, and "
            "an FSM-based controller with bit-reversal addressing."
        ),
        "hw_fw_sw_partitioning": (
            "All-hardware implementation. No firmware or software component. "
            "The design is a fully synchronous RTL datapath with no CPU or bus master."
        ),
        "system_flows": [
            {
                "name": "fft_pipeline",
                "description": (
                    "Input samples arrive via AXI-Stream, pass through 4 butterfly "
                    "stages with twiddle factor lookup, and exit as frequency-domain "
                    "coefficients in bit-reversed order."
                ),
            },
        ],
        "technology_rationale": (
            "sky130 selected for open-source PDK availability. 130nm process provides "
            "sufficient speed for 50 MHz target with comfortable timing margin."
        ),
        "architecture_decisions": [
            {
                "decision": "Pipeline vs batch processing",
                "chosen": "Pipeline",
                "rationale": (
                    "Streaming pipeline achieves one sample per clock throughput "
                    "without requiring large SRAM buffers, minimizing area."
                ),
            },
            {
                "decision": "Twiddle factor storage",
                "chosen": "On-chip ROM",
                "rationale": (
                    "8-entry ROM for 16-point FFT is small (~800 gates). External "
                    "memory would add latency and interface complexity."
                ),
            },
        ],
        "risk_assessment": [
            {
                "risk": "Timing closure at 50 MHz",
                "severity": "low",
                "mitigation": (
                    "130nm process has ~2 GHz fT; 50 MHz is well within margin. "
                    "Butterfly multiplier is the critical path."
                ),
            },
            {
                "risk": "Area budget overshoot",
                "severity": "medium",
                "mitigation": (
                    "Estimated 4,300 gates vs 50,000 budget. Large margin available."
                ),
            },
        ],
    },
    "phase": "sad_complete",
}

# Markdown-format SAD (new format -- LLM produces markdown directly)
FFT16_SAD_MARKDOWN = {
    "sad_text": """\
# SAD — 16-Point FFT Processor

## System Overview
Single-chip radix-2 DIT FFT processor targeting sky130 at 50 MHz. \
Composed of a 4-stage butterfly pipeline, twiddle factor ROM, and \
an FSM-based controller with bit-reversal addressing.

## HW/FW/SW Partitioning
All-hardware implementation. No firmware or software component. \
The design is a fully synchronous RTL datapath with no CPU or bus master.

## System Flows
### fft_pipeline
Input samples arrive via AXI-Stream, pass through 4 butterfly \
stages with twiddle factor lookup, and exit as frequency-domain \
coefficients in bit-reversed order.

## Technology Rationale
sky130 selected for open-source PDK availability. 130nm process provides \
sufficient speed for 50 MHz target with comfortable timing margin.

## Architecture Decisions
- **Pipeline vs batch processing**: Pipeline -- Streaming pipeline achieves \
one sample per clock throughput without requiring large SRAM buffers.
- **Twiddle factor storage**: On-chip ROM -- 8-entry ROM for 16-point FFT \
is small (~800 gates).

## Risk Assessment
- **Timing closure at 50 MHz** (low): 130nm process has ~2 GHz fT; 50 MHz \
is well within margin. Butterfly multiplier is the critical path.
- **Area budget overshoot** (medium): Estimated 4,300 gates vs 50,000 budget.
""",
    "phase": "sad_complete",
}

# ═══════════════════════════════════════════════════════════════════════════
# FRD (Functional Requirements Document) Mock Output
# ═══════════════════════════════════════════════════════════════════════════

FFT16_FRD_DOCUMENT = {
    "frd": {
        "title": "16-Point FFT Processor FRD",
        "performance_requirements": [
            {
                "id": "PERF-001",
                "requirement": "Sustain 50 Msps throughput",
                "acceptance_criteria": "One complex sample per clock at 50 MHz",
            },
            {
                "id": "PERF-002",
                "requirement": "Latency below 320 ns",
                "acceptance_criteria": "16 clock cycles at 50 MHz (0.32 us)",
            },
        ],
        "interface_requirements": [
            {
                "id": "IF-001",
                "requirement": "AXI-Stream input interface",
                "acceptance_criteria": "TVALID/TREADY/TDATA[31:0] (16-bit I + 16-bit Q)",
            },
            {
                "id": "IF-002",
                "requirement": "AXI-Stream output interface",
                "acceptance_criteria": "TVALID/TREADY/TDATA[31:0] with TLAST on final bin",
            },
        ],
        "timing_requirements": [
            {
                "id": "TIM-001",
                "requirement": "Single clock domain at 50 MHz",
                "acceptance_criteria": "All registers clocked by clk_sys, no CDC",
            },
        ],
        "resource_budgets": {
            "area": {"max_gate_count": 50000, "notes": "Butterfly dominated (~2000 gates)"},
            "power": {"total_power_mw": 100, "notes": "Not critical for this design"},
        },
        "testability_requirements": [
            {
                "id": "TEST-001",
                "requirement": "Bit-exact match with Python golden model",
                "acceptance_criteria": "cocotb testbench compares RTL output against PyDVB/dvb/OFDM.py",
            },
        ],
    },
    "phase": "frd_complete",
}

# Markdown-format FRD (new format -- LLM produces markdown directly)
FFT16_FRD_MARKDOWN = {
    "frd_text": """\
# FRD — 16-Point FFT Processor

## Performance Requirements
- **PERF-001**: Sustain 50 Msps throughput -- One complex sample per clock at 50 MHz (must_have)
- **PERF-002**: Latency below 320 ns -- 16 clock cycles at 50 MHz (must_have)

## Interface Requirements
- **IF-001**: AXI-Stream input -- TVALID/TREADY/TDATA[31:0] 16-bit I + 16-bit Q (must_have)
- **IF-002**: AXI-Stream output -- TVALID/TREADY/TDATA[31:0] with TLAST on final bin (must_have)

## Timing Requirements
- **TIM-001**: Single clock domain at 50 MHz -- All registers clocked by clk_sys, no CDC (must_have)

## Resource Budgets
### Area
- **max_gate_count:** 50000
- **notes:** Butterfly dominated (~2000 gates)

### Power
- **total_power_mw:** 100
- **notes:** Not critical for this design

## Testability Requirements
- Bit-exact match with Python golden model (cocotb testbench compares RTL output against PyDVB/dvb/OFDM.py)
""",
    "phase": "frd_complete",
}

# ═══════════════════════════════════════════════════════════════════════════
# Memory Map / Clock Tree / Register Spec
# ═══════════════════════════════════════════════════════════════════════════

FFT16_MEMORY_MAP = {
    "stub": True, "questions": [],
    "result": {
        "sram": {"base_address": "0x00000000", "base_address_int": 0, "size": 0x8000, "size_kb": 32},
        "peripherals": [
            {"name": "fft_butterfly", "base_address": "0x10000000", "base_address_int": 0x10000000, "size": 0x100},
            {"name": "twiddle_rom", "base_address": "0x20000000", "base_address_int": 0x20000000, "size": 0x100},
            {"name": "fft_controller", "base_address": "0x30000000", "base_address_int": 0x30000000, "size": 0x100},
        ],
        "top_csr": {"base_address": "0x80000000", "base_address_int": 0x80000000, "size": 0x100},
    },
}

FFT16_CLOCK_TREE = {
    "stub": True, "questions": [],
    "result": {
        "domains": [{"name": "clk_sys", "frequency_mhz": 50.0, "source": "PLL or external"}],
        "crossings": [],
        "reset_spec": {"strategy": "synchronous", "domains": ["clk_sys"]},
        "num_domains": 1, "cdc_required": False,
    },
}

FFT16_REGISTER_SPEC = {
    "stub": True, "questions": [],
    "result": {
        "total_blocks": 4,
        "blocks": [
            {"name": "fft_butterfly", "num_config": 8, "num_status": 8},
            {"name": "twiddle_rom", "num_config": 8, "num_status": 8},
            {"name": "fft_controller", "num_config": 8, "num_status": 8},
            {"name": "top_csr", "num_config": 8, "num_status": 8},
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Constraint Check Results
# ═══════════════════════════════════════════════════════════════════════════

FFT16_CONSTRAINT_PASS = {
    "violations": [], "all_pass": True, "has_structural": False, "total_gate_estimate": 4300,
}

FFT16_CONSTRAINT_STRUCTURAL = {
    "violations": [{"violation": "Peripheral count exceeds 8-slot nibble decoder",
                    "severity": "error", "category": "structural"}],
    "all_pass": False, "has_structural": True,
}

FFT16_CONSTRAINT_AUTO_FIXABLE = {
    "violations": [{"violation": "Gate budget exceeded by 5% (52,500 vs 50,000 limit)",
                    "severity": "warning", "category": "auto_fixable"}],
    "all_pass": False, "has_structural": False,
}
