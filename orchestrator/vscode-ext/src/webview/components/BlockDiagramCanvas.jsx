import React, { useState, useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  Controls,
  MiniMap,
  Background,
  MarkerType,
  useNodesState,
  useEdgesState,
  getBezierPath,
  Handle,
  Position,
} from 'reactflow';
import { applyElkLayout } from '../utils/layout';

// ---------------------------------------------------------------------------
// Node type styles -- matching taskgraph_dash_component exactly
// ---------------------------------------------------------------------------

const TYPE_COLORS = {
  compute:      'rgba(128, 170, 255, 0.8)',    // Blue
  hwa:          'rgba(48, 130, 96, 0.8)',       // Green
  bus:          'rgb(137, 120, 191)',            // Purple
  memory:       'rgba(212, 92, 67, 0.8)',       // Red-orange
  sensor:       'rgba(50, 205, 205, 0.8)',      // Cyan
  power_domain: 'rgba(255, 246, 66, 0.8)',      // Yellow
  pll:          'rgba(100, 200, 150, 0.8)',      // Teal-green
  pcie:         'rgba(160, 120, 200, 0.8)',      // Lavender
  i3c:          'rgba(200, 160, 80, 0.8)',       // Gold
  gpio:         'rgba(170, 170, 170, 0.8)',      // Gray
  pmic:         'rgba(156, 244, 174, 0.8)',      // Light green
};

const TYPE_ICONS = {
  compute: '\u2699',      // gear
  hwa: '\u26A1',          // lightning
  bus: '\u2194',          // left-right arrow
  memory: '\u25A6',       // filled square with lines
  sensor: '\u25CE',       // bullseye
  power_domain: '\u26A0', // warning (power)
  pll: '\u223F',          // sine wave
  pcie: '\u2550',         // double horizontal
  i3c: '\u2505',          // dash
  gpio: '\u25A3',         // filled square
  pmic: '\u2607',         // lightning
};

function getTypeColor(nodeType) {
  if (nodeType in TYPE_COLORS) return TYPE_COLORS[nodeType];
  if (!nodeType) return 'rgba(234, 234, 234, 0.86)';
  // Dynamic fallback (same as taskgraph_dash_component)
  const len = nodeType.length;
  const r = (len * 10) % 256;
  const g = (nodeType.charCodeAt(0) * 5) % 256;
  const b = (nodeType.charCodeAt(len - 1) * 5) % 256;
  return `rgba(${r}, ${g}, ${b}, 0.6)`;
}


// ---------------------------------------------------------------------------
// Custom Architecture Node -- matching taskgraph NodeArchGraph
// ---------------------------------------------------------------------------

function ArchBlockNode({ data, selected }) {
  const isGroup = data.is_subsystem || data.node_type === '' || data.node_type === 'power_domain';
  const isBus = data.node_type === 'bus';
  const color = getTypeColor(data.node_type);
  const icon = TYPE_ICONS[data.node_type] || '\u25CF';

  if (isGroup) {
    const isPowerDomain = data.node_type === 'power_domain';
    return (
      <div
        data-testid="group-node"
        style={{
          border: selected ? '2px solid #ff0000' : '2px dashed var(--color-border-primary)',
          borderRadius: 8,
          boxShadow: 'var(--shadow-card)',
          backgroundColor: isPowerDomain ? 'rgba(242, 128, 70, 0.2)' : 'var(--color-diagram-group-bg)',
          padding: 5,
          width: '100%',
          height: '100%',
          minWidth: 200,
          minHeight: 100,
        }}
      >
        <div
          style={{
            backgroundColor: isPowerDomain ? 'rgba(242, 128, 70, 0.6)' : 'var(--color-diagram-group-header)',
            borderRadius: 5,
            padding: '6px 12px',
            minHeight: 20,
            whiteSpace: 'normal',
            borderBottom: '1px dashed var(--color-border-primary)',
          }}
        >
          <b style={{
            fontSize: '11pt',
            textTransform: 'uppercase',
            letterSpacing: 0.8,
            color: 'var(--color-diagram-group-text)',
            textShadow: '1px 1px 2px rgba(255,255,255,0.5)',
          }}>
            {data.node_name}
          </b>
        </div>
      </div>
    );
  }

  if (isBus) {
    // Bus hub node -- matching taskgraph_dash_component bus styling:
    // horizontal bar with triangular arrow caps on both ends, purple theme.
    return (
      <div data-testid="bus-node">
        <Handle type="target" position={Position.Top} />
        <Handle type="target" position={Position.Left} id="left-in" />
        <div
          className="block-diagram-bus-header"
          style={{
            border: '1px solid rgba(0,0,0,0.1)',
            borderRadius: 5,
            padding: '8px 30px',
            backgroundColor: color,
            minWidth: 200,
            minHeight: 60,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 18, color: '#fff' }}>{icon}</span>
            <b style={{
              fontSize: '13pt',
              textShadow: '1px 1px 2px rgba(0,0,0,0.2)',
              color: '#fff',
            }}>
              {data.device_name || data.node_name}
            </b>
          </div>
          {data.node_notes && (
            <div style={{
              marginTop: 4, fontSize: 10, color: 'rgba(255,255,255,0.8)',
              textAlign: 'center', lineHeight: 1.3,
            }}>
              {data.node_notes}
            </div>
          )}
        </div>
        <Handle type="source" position={Position.Bottom} />
        <Handle type="source" position={Position.Right} id="right-out" />
      </div>
    );
  }

  // Regular block node
  return (
    <div data-testid="block-node">
      <Handle type="target" position={Position.Top} />
      <div
        style={{
          border: selected ? '2px solid #ffba00' : '1px solid var(--color-diagram-node-border)',
          borderRadius: 5,
          boxShadow: 'var(--shadow-card)',
          backgroundColor: selected ? 'var(--color-diagram-selected-bg)' : 'var(--color-diagram-node-bg)',
          textAlign: 'left',
          width: 340,
          minHeight: 160,
        }}
      >
        {/* Header */}
        <div
          style={{
            border: '1px solid rgba(0,0,0,0.1)',
            borderRadius: 5,
            padding: '5px 10px',
            boxShadow: `0 4px 8px ${color.replace(/[\d.]+\)$/, '0.3)')}`,
            backgroundColor: color,
            width: '100%',
            whiteSpace: 'normal',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 16 }}>{icon}</span>
              <b style={{
                paddingTop: 5, paddingBottom: 5,
                fontSize: '13pt',
                textShadow: '1px 1px 2px rgba(0,0,0,0.2)',
                color: '#fff',
              }}>
                {data.device_name || data.node_name}
              </b>
            </div>
            <span style={{
              fontSize: 9, fontWeight: 600,
              textTransform: 'uppercase',
              padding: '2px 6px',
              borderRadius: 3,
              background: 'rgba(255,255,255,0.3)',
              color: '#fff',
              letterSpacing: 0.5,
            }}>
              {data.node_type}
            </span>
          </div>
        </div>

        {/* Body */}
        <div style={{ textAlign: 'left', padding: '8px 10px', fontSize: 12, color: 'var(--color-diagram-body-text)', lineHeight: 1.6 }}>
          {data.frequency && (
            <div><b>Frequency: </b>{data.frequency}</div>
          )}
          {data.module_type && data.module_type !== data.node_type && (
            <div><b>Module: </b>{data.module_type}</div>
          )}
          {data.node_notes && (
            <div style={{ marginTop: 4, fontSize: 11, color: 'var(--color-diagram-body-muted)', lineHeight: 1.4 }}>
              {data.node_notes}
            </div>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}


// ---------------------------------------------------------------------------
// Custom Architecture Edge -- matching taskgraph edgeArchGraph
// ---------------------------------------------------------------------------

function ArchBlockEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, selected }) {
  const [edgePath] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const isBusEdge = data?.connection_type === 'bus_connect';

  // Bus edges use purple (#8978bf) to match the bus node color;
  // regular edges use green (#22DD22) matching taskgraph_dash_component.
  const strokeColor = isBusEdge
    ? (selected ? '#a090d0' : '#8978bf')
    : (selected ? '#44FF44' : '#22DD22');

  return (
    <>
      <path
        id={id}
        d={edgePath}
        data-testid={isBusEdge ? 'bus-edge' : 'direct-edge'}
        style={{
          stroke: strokeColor,
          strokeWidth: selected ? 4 : (isBusEdge ? 4 : 3),
          strokeDasharray: selected ? '5 5' : 'none',
          fill: 'none',
        }}
        markerEnd={isBusEdge ? 'url(#bus-arrow)' : 'url(#arch-arrow)'}
      />
      {data?.label && (
        <text>
          <textPath
            href={`#${id}`}
            startOffset="50%"
            textAnchor="middle"
            style={{ fontSize: 10, fill: isBusEdge ? '#8978bf' : '#666' }}
          >
            {data.label}
          </textPath>
        </text>
      )}
    </>
  );
}


// ---------------------------------------------------------------------------
// Node / edge type registrations
// ---------------------------------------------------------------------------

const nodeTypes = { nodeArchGraph: ArchBlockNode };
const edgeTypes = { edgeArchGraph: ArchBlockEdge };

// Default edge options
const defaultEdgeOptions = {
  type: 'edgeArchGraph',
  markerEnd: {
    type: MarkerType.ArrowClosed,
    strokeWidth: 3,
    width: 30,
  },
};


// ---------------------------------------------------------------------------
// Main BlockDiagramCanvas
// ---------------------------------------------------------------------------

export default function BlockDiagramCanvas({ diagramData }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [layoutDone, setLayoutDone] = useState(false);

  // Transform the diagram JSON into ReactFlow nodes/edges and apply layout
  useEffect(() => {
    if (!diagramData?.architecture) return;

    const arch = diagramData.architecture;
    const rawNodes = (arch.systemNodes || []).map((n) => ({
      ...n,
      type: n.type || 'nodeArchGraph',
      position: n.position || { x: 0, y: 0 },
      style: n.data?.is_subsystem
        ? { width: 600, height: 400 }
        : { width: 340, minHeight: 160 },
      ...(n.data?.is_subsystem && n.data?.node_parentId !== 'graph_root'
        ? { parentId: n.data.node_parentId, extent: 'parent' }
        : {}),
    }));

    const rawEdges = (arch.systemEdges || []).map((e) => ({
      ...e,
      type: e.type || 'edgeArchGraph',
    }));

    // Apply ELK layout
    applyElkLayout(rawNodes, rawEdges, 'DOWN')
      .then((laid) => {
        setNodes(laid);
        setEdges(rawEdges);
        setLayoutDone(true);
      })
      .catch(() => {
        // Fallback: use raw positions
        setNodes(rawNodes);
        setEdges(rawEdges);
        setLayoutDone(true);
      });
  }, [diagramData, setNodes, setEdges]);

  // Design name header
  const designName = diagramData?.architecture?.designName || diagramData?.metadata?.design_name || '';
  const blockCount = diagramData?.metadata?.block_count || nodes.length;
  const connCount = diagramData?.metadata?.connection_count || edges.length;

  // Count bus nodes and subsystem groups for the info overlay
  const busCount = useMemo(() => {
    const sysNodes = diagramData?.architecture?.systemNodes || [];
    return sysNodes.filter(n => n.data?.node_type === 'bus').length;
  }, [diagramData]);
  const subsystemCount = useMemo(() => {
    const sysNodes = diagramData?.architecture?.systemNodes || [];
    return sysNodes.filter(n => n.data?.is_subsystem).length;
  }, [diagramData]);

  if (!diagramData) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'var(--color-text-placeholder)', fontSize: 14,
      }}>
        No block diagram data available. Run the architecture graph to generate.
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Design info overlay */}
      <div style={{
        position: 'absolute', top: 12, left: 16, zIndex: 10,
        background: 'var(--color-diagram-overlay)',
        border: '1px solid var(--color-diagram-overlay-border)', borderRadius: 8,
        padding: '8px 16px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
        backdropFilter: 'blur(8px)',
        fontSize: 12,
        fontFamily: "'Roboto', sans-serif",
      }}>
        <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text-primary)', marginBottom: 2 }}>
          {designName}
        </div>
        <div style={{ color: 'var(--color-text-dimmed)', fontSize: 11 }}>
          {blockCount} blocks &middot; {connCount} connections
          {busCount > 0 && <> &middot; {busCount} {busCount === 1 ? 'bus' : 'buses'}</>}
          {subsystemCount > 0 && <> &middot; {subsystemCount} {subsystemCount === 1 ? 'subsystem' : 'subsystems'}</>}
        </div>
      </div>

      {/* Arrow marker definitions */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <marker
            id="arch-arrow"
            viewBox="0 0 10 10"
            refX="8" refY="5"
            markerWidth="8" markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#22DD22" />
          </marker>
          <marker
            id="bus-arrow"
            viewBox="0 0 10 10"
            refX="8" refY="5"
            markerWidth="8" markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8978bf" />
          </marker>
        </defs>
      </svg>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="var(--color-diagram-bg)" gap={20} />
        <Controls position="bottom-right" />
        <MiniMap
          nodeStrokeWidth={3}
          nodeColor={(node) => {
            const nt = node.data?.node_type;
            return TYPE_COLORS[nt] || '#ddd';
          }}
          style={{ border: '1px solid #ccc', borderRadius: 8 }}
        />
      </ReactFlow>
    </div>
  );
}
