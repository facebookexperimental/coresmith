import React from 'react';
import { Handle, Position } from 'reactflow';

export default function TerminalNode({ data, selected }) {
  const targetPos = data.direction === 'RIGHT' ? Position.Left : Position.Top;
  const sourcePos = data.direction === 'RIGHT' ? Position.Right : Position.Bottom;

  return (
    <div className={`workflow-node terminal-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={targetPos} />
      <div className="node-header terminal-header">
        <span className="node-icon">⏹</span>
        <span className="node-title">{data.label}</span>
        {data.status === 'running' && <span className="active-badge">Working</span>}
      </div>
      {data.description && (
        <div className="node-body">
          <div className="node-description">{data.description}</div>
        </div>
      )}
      <Handle type="source" position={sourcePos} />
    </div>
  );
}
