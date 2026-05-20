import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import BlockDiagramCanvas from '../BlockDiagramCanvas';

// ReactFlow requires a container with dimensions for rendering.
// Mock getBoundingClientRect so ReactFlow doesn't bail on zero-size container.
beforeAll(() => {
  Element.prototype.getBoundingClientRect = jest.fn(() => ({
    width: 1200,
    height: 800,
    top: 0,
    left: 0,
    bottom: 800,
    right: 1200,
    x: 0,
    y: 0,
    toJSON: () => {},
  }));

  // ReactFlow uses ResizeObserver internally
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };

  // ReactFlow uses IntersectionObserver for viewport detection
  global.IntersectionObserver = class {
    constructor() {}
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});


// ---------------------------------------------------------------------------
// Test data fixtures
// ---------------------------------------------------------------------------

/** Minimal diagram with no blocks -- tests empty/fallback states */
const EMPTY_DIAGRAM = null;

/** Simple diagram: two compute blocks with a direct connection */
const SIMPLE_DIAGRAM = {
  version: { id: 'reactflow_json_1.0.0' },
  metadata: {
    design_name: 'Test Design',
    timestamp: '2025-01-01T00:00:00Z',
    source: 'socmate_architecture',
    block_count: 2,
    connection_count: 1,
  },
  architecture: {
    designName: 'Test Design',
    systemNodes: [
      {
        id: 'encoder',
        type: 'nodeArchGraph',
        position: { x: 0, y: 0 },
        data: {
          node_type: 'compute',
          node_name: 'encoder',
          node_parentId: 'graph_root',
          connect_to: [{ absolute_path: 'decoder', local_path: 'decoder' }],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'encoder',
          frequency: '50 MHz',
          module_type: 'compute',
          node_notes: 'Encodes data | Tier 1 | ~5K gates',
        },
      },
      {
        id: 'decoder',
        type: 'nodeArchGraph',
        position: { x: 0, y: 200 },
        data: {
          node_type: 'compute',
          node_name: 'decoder',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'decoder',
          frequency: '50 MHz',
          module_type: 'compute',
          node_notes: 'Decodes data | Tier 1 | ~5K gates',
        },
      },
    ],
    systemEdges: [
      {
        id: 'eencoder-decoder',
        source: 'encoder',
        target: 'decoder',
        type: 'edgeArchGraph',
        data: {
          label: 'encoder->decoder AXI-Stream 8b',
          connection_type: 'depends_on',
        },
      },
    ],
    systemLayout: {
      elk_layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
      },
    },
    moduleTypeOptions: ['compute', 'bus', 'memory'],
  },
};

/** Diagram with a bus hub node and star-topology edges */
const BUS_DIAGRAM = {
  version: { id: 'reactflow_json_1.0.0' },
  metadata: {
    design_name: 'Bus Test Design',
    timestamp: '2025-01-01T00:00:00Z',
    source: 'socmate_architecture',
    block_count: 3,
    connection_count: 4,
  },
  architecture: {
    designName: 'Bus Test Design',
    systemNodes: [
      {
        id: 'cpu',
        type: 'nodeArchGraph',
        position: { x: 0, y: 0 },
        data: {
          node_type: 'compute',
          node_name: 'cpu',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'cpu',
          frequency: '100 MHz',
          module_type: 'compute',
          node_notes: 'Main processor',
        },
      },
      {
        id: 'bus__axi_interconnect',
        type: 'nodeArchGraph',
        position: { x: 200, y: 100 },
        data: {
          node_type: 'bus',
          node_name: 'axi_interconnect',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'axi_interconnect',
          frequency: '',
          module_type: 'bus',
          node_notes: 'AXI4 | Width: 32b | 3 ports',
        },
      },
      {
        id: 'sram',
        type: 'nodeArchGraph',
        position: { x: 0, y: 200 },
        data: {
          node_type: 'memory',
          node_name: 'sram',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'sram',
          frequency: '100 MHz',
          module_type: 'memory',
          node_notes: '64KB SRAM',
        },
      },
      {
        id: 'uart',
        type: 'nodeArchGraph',
        position: { x: 400, y: 200 },
        data: {
          node_type: 'hwa',
          node_name: 'uart',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'uart',
          frequency: '100 MHz',
          module_type: 'hwa',
          node_notes: 'UART peripheral',
        },
      },
    ],
    systemEdges: [
      {
        id: 'ecpu-axi_interconnect',
        source: 'cpu',
        target: 'bus__axi_interconnect',
        type: 'edgeArchGraph',
        data: {
          label: 'cpu->axi_interconnect AXI4 32b',
          connection_type: 'bus_connect',
        },
      },
      {
        id: 'eaxi_interconnect-sram',
        source: 'bus__axi_interconnect',
        target: 'sram',
        type: 'edgeArchGraph',
        data: {
          label: 'axi_interconnect->sram AXI4 32b',
          connection_type: 'bus_connect',
        },
      },
      {
        id: 'eaxi_interconnect-uart',
        source: 'bus__axi_interconnect',
        target: 'uart',
        type: 'edgeArchGraph',
        data: {
          label: 'axi_interconnect->uart AXI4 32b',
          connection_type: 'bus_connect',
        },
      },
    ],
    systemLayout: {
      elk_layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
      },
    },
    moduleTypeOptions: ['compute', 'bus', 'memory', 'hwa'],
  },
};

/** Diagram with subsystem grouping */
const SUBSYSTEM_DIAGRAM = {
  version: { id: 'reactflow_json_1.0.0' },
  metadata: {
    design_name: 'Subsystem Test Design',
    timestamp: '2025-01-01T00:00:00Z',
    source: 'socmate_architecture',
    block_count: 3,
    connection_count: 2,
  },
  architecture: {
    designName: 'Subsystem Test Design',
    systemNodes: [
      {
        id: 'encode_pipeline',
        type: 'nodeArchGraph',
        position: { x: 0, y: 0 },
        data: {
          node_type: '',
          node_name: 'encode_pipeline',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: true,
          is_power_domain: false,
          device_name: 'encode_pipeline',
          frequency: '',
          module_type: '',
          node_notes: 'Subsystem: encode_pipeline',
        },
      },
      {
        id: 'encode_pipeline.dct',
        type: 'nodeArchGraph',
        position: { x: 0, y: 0 },
        data: {
          node_type: 'compute',
          node_name: 'encode_pipeline.dct',
          node_parentId: 'encode_pipeline',
          connect_to: [{ absolute_path: 'encode_pipeline.quantizer', local_path: 'quantizer' }],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'dct',
          frequency: '50 MHz',
          module_type: 'compute',
          node_notes: 'DCT transform | Tier 2',
        },
      },
      {
        id: 'encode_pipeline.quantizer',
        type: 'nodeArchGraph',
        position: { x: 0, y: 200 },
        data: {
          node_type: 'compute',
          node_name: 'encode_pipeline.quantizer',
          node_parentId: 'encode_pipeline',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'quantizer',
          frequency: '50 MHz',
          module_type: 'compute',
          node_notes: 'Quantizer | Tier 1',
        },
      },
    ],
    systemEdges: [
      {
        id: 'edct-quantizer',
        source: 'encode_pipeline.dct',
        target: 'encode_pipeline.quantizer',
        type: 'edgeArchGraph',
        data: {
          label: 'dct->quantizer AXI-Stream 8b',
          connection_type: 'depends_on',
        },
      },
    ],
    systemLayout: {
      elk_layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
      },
    },
    moduleTypeOptions: ['compute'],
  },
};

/** Combined: bus + subsystems + direct connections */
const FULL_DIAGRAM = {
  version: { id: 'reactflow_json_1.0.0' },
  metadata: {
    design_name: 'Full Architecture Test',
    timestamp: '2025-01-01T00:00:00Z',
    source: 'socmate_architecture',
    block_count: 5,
    connection_count: 6,
  },
  architecture: {
    designName: 'Full Architecture Test',
    systemNodes: [
      // Subsystem group
      {
        id: 'datapath',
        type: 'nodeArchGraph',
        position: { x: 0, y: 0 },
        data: {
          node_type: '',
          node_name: 'datapath',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: true,
          is_power_domain: false,
          device_name: 'datapath',
          frequency: '',
          module_type: '',
          node_notes: 'Subsystem: datapath',
        },
      },
      // Child blocks in subsystem
      {
        id: 'datapath.encoder',
        type: 'nodeArchGraph',
        position: { x: 0, y: 0 },
        data: {
          node_type: 'compute',
          node_name: 'datapath.encoder',
          node_parentId: 'datapath',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'encoder',
          frequency: '50 MHz',
          module_type: 'compute',
          node_notes: 'Encoder block',
        },
      },
      {
        id: 'datapath.decoder',
        type: 'nodeArchGraph',
        position: { x: 0, y: 200 },
        data: {
          node_type: 'compute',
          node_name: 'datapath.decoder',
          node_parentId: 'datapath',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'decoder',
          frequency: '50 MHz',
          module_type: 'compute',
          node_notes: 'Decoder block',
        },
      },
      // Bus node
      {
        id: 'bus__apb_bus',
        type: 'nodeArchGraph',
        position: { x: 400, y: 100 },
        data: {
          node_type: 'bus',
          node_name: 'apb_bus',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'apb_bus',
          frequency: '',
          module_type: 'bus',
          node_notes: 'APB | Width: 32b | 2 ports',
        },
      },
      // Standalone block
      {
        id: 'csr_bridge',
        type: 'nodeArchGraph',
        position: { x: 400, y: 300 },
        data: {
          node_type: 'hwa',
          node_name: 'csr_bridge',
          node_parentId: 'graph_root',
          connect_to: [],
          is_subsystem: false,
          is_power_domain: false,
          device_name: 'csr_bridge',
          frequency: '50 MHz',
          module_type: 'hwa',
          node_notes: 'CSR register bridge',
        },
      },
    ],
    systemEdges: [
      // Direct connection within subsystem
      {
        id: 'eencoder-decoder',
        source: 'datapath.encoder',
        target: 'datapath.decoder',
        type: 'edgeArchGraph',
        data: {
          label: 'encoder->decoder AXI-Stream 8b',
          connection_type: 'depends_on',
        },
      },
      // Bus connections
      {
        id: 'eencoder-apb_bus',
        source: 'datapath.encoder',
        target: 'bus__apb_bus',
        type: 'edgeArchGraph',
        data: {
          label: 'encoder->apb_bus APB 32b',
          connection_type: 'bus_connect',
        },
      },
      {
        id: 'eapb_bus-csr_bridge',
        source: 'bus__apb_bus',
        target: 'csr_bridge',
        type: 'edgeArchGraph',
        data: {
          label: 'apb_bus->csr_bridge APB 32b',
          connection_type: 'bus_connect',
        },
      },
    ],
    systemLayout: {
      elk_layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'DOWN',
      },
    },
    moduleTypeOptions: ['compute', 'bus', 'hwa'],
  },
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('BlockDiagramCanvas', () => {
  it('renders empty state when diagramData is null', () => {
    const { container } = render(<BlockDiagramCanvas diagramData={null} />);
    expect(container.textContent).toContain('No block diagram data available');
  });

  it('renders without crashing with a simple two-block diagram', async () => {
    const { container } = render(<BlockDiagramCanvas diagramData={SIMPLE_DIAGRAM} />);

    // Design name should appear in the info overlay
    expect(container.textContent).toContain('Test Design');
    expect(container.textContent).toContain('2 blocks');
    expect(container.textContent).toContain('1 connections');
  });

  it('renders bus hub nodes with the bus-specific class', async () => {
    const { container } = render(<BlockDiagramCanvas diagramData={BUS_DIAGRAM} />);

    // Design name
    expect(container.textContent).toContain('Bus Test Design');

    // Wait for layout to complete and nodes to render
    await waitFor(() => {
      expect(container.textContent).toContain('axi_interconnect');
    });

    // Bus node should get the block-diagram-bus-header CSS class
    const busHeaders = container.querySelectorAll('.block-diagram-bus-header');
    expect(busHeaders.length).toBeGreaterThanOrEqual(1);

    // Should show bus count in the overlay
    expect(container.textContent).toContain('1 bus');
  });

  it('renders subsystem group nodes', async () => {
    const { container } = render(<BlockDiagramCanvas diagramData={SUBSYSTEM_DIAGRAM} />);

    expect(container.textContent).toContain('Subsystem Test Design');

    await waitFor(() => {
      expect(container.textContent).toContain('encode_pipeline');
    });

    // Should show subsystem count
    expect(container.textContent).toContain('1 subsystem');
  });

  it('does not crash with combined bus + subsystem + direct topology', async () => {
    // This is the critical "does not crash" test with all features combined
    const { container } = render(<BlockDiagramCanvas diagramData={FULL_DIAGRAM} />);

    expect(container.textContent).toContain('Full Architecture Test');
    expect(container.textContent).toContain('5 blocks');

    await waitFor(() => {
      expect(container.textContent).toContain('apb_bus');
    });

    // Should show both bus and subsystem counts
    expect(container.textContent).toContain('1 bus');
    expect(container.textContent).toContain('1 subsystem');
  });

  it('renders both SVG marker definitions (arch-arrow and bus-arrow)', () => {
    const { container } = render(<BlockDiagramCanvas diagramData={SIMPLE_DIAGRAM} />);

    const archArrow = container.querySelector('#arch-arrow');
    const busArrow = container.querySelector('#bus-arrow');
    expect(archArrow).toBeInTheDocument();
    expect(busArrow).toBeInTheDocument();
  });

  it('handles diagram with zero edges gracefully', async () => {
    const noEdgeDiagram = {
      ...SIMPLE_DIAGRAM,
      architecture: {
        ...SIMPLE_DIAGRAM.architecture,
        systemEdges: [],
      },
      metadata: {
        ...SIMPLE_DIAGRAM.metadata,
        connection_count: 0,
      },
    };

    const { container } = render(<BlockDiagramCanvas diagramData={noEdgeDiagram} />);
    expect(container.textContent).toContain('0 connections');
  });

  it('handles diagram with only bus nodes (no regular blocks)', async () => {
    const busOnlyDiagram = {
      version: { id: 'reactflow_json_1.0.0' },
      metadata: {
        design_name: 'Bus Only',
        timestamp: '2025-01-01T00:00:00Z',
        source: 'socmate_architecture',
        block_count: 1,
        connection_count: 0,
      },
      architecture: {
        designName: 'Bus Only',
        systemNodes: [
          {
            id: 'bus__main',
            type: 'nodeArchGraph',
            position: { x: 0, y: 0 },
            data: {
              node_type: 'bus',
              node_name: 'main_bus',
              node_parentId: 'graph_root',
              connect_to: [],
              is_subsystem: false,
              is_power_domain: false,
              device_name: 'main_bus',
              frequency: '',
              module_type: 'bus',
              node_notes: 'AXI4 | 4 ports',
            },
          },
        ],
        systemEdges: [],
        systemLayout: {
          elk_layoutOptions: {
            'elk.algorithm': 'layered',
            'elk.direction': 'DOWN',
          },
        },
        moduleTypeOptions: ['bus'],
      },
    };

    const { container } = render(<BlockDiagramCanvas diagramData={busOnlyDiagram} />);
    expect(container.textContent).toContain('Bus Only');
    expect(container.textContent).toContain('1 bus');
  });

  it('shows pluralized counts correctly', async () => {
    // Two buses
    const twoBusDiagram = {
      ...BUS_DIAGRAM,
      architecture: {
        ...BUS_DIAGRAM.architecture,
        systemNodes: [
          ...BUS_DIAGRAM.architecture.systemNodes,
          {
            id: 'bus__apb_bus',
            type: 'nodeArchGraph',
            position: { x: 400, y: 100 },
            data: {
              node_type: 'bus',
              node_name: 'apb_bus',
              node_parentId: 'graph_root',
              connect_to: [],
              is_subsystem: false,
              is_power_domain: false,
              device_name: 'apb_bus',
              frequency: '',
              module_type: 'bus',
              node_notes: 'APB | 2 ports',
            },
          },
        ],
      },
    };

    const { container } = render(<BlockDiagramCanvas diagramData={twoBusDiagram} />);
    expect(container.textContent).toContain('2 buses');
  });

  it('does not show bus/subsystem counts when there are none', () => {
    const { container } = render(<BlockDiagramCanvas diagramData={SIMPLE_DIAGRAM} />);
    expect(container.textContent).not.toContain('bus');
    expect(container.textContent).not.toContain('subsystem');
  });

  it('handles missing architecture field without crashing', () => {
    const badDiagram = {
      version: { id: 'reactflow_json_1.0.0' },
      metadata: { design_name: 'Bad', source: 'test' },
    };
    // Should render the empty state since architecture is missing
    const { container } = render(<BlockDiagramCanvas diagramData={badDiagram} />);
    // Should not crash -- just render nothing useful
    expect(container).toBeTruthy();
  });
});
