import React from 'react';
import { Handle, Position } from 'reactflow';

export default function ConditionNode({ data, selected }) {
  const targetPos = data.direction === 'RIGHT' ? Position.Left : Position.Top;
  const sourcePos = data.direction === 'RIGHT' ? Position.Right : Position.Bottom;

  return (
    <div className={`workflow-node condition-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={targetPos} />
      <div className="node-header condition-header">
        <span className="node-icon">◆</span>
        <span className="node-title">{data.label}</span>
      </div>
      <div className="node-body">
        <div className="node-function">{data.function}</div>
      </div>
      <Handle type="source" position={sourcePos} />
    </div>
  );
}
