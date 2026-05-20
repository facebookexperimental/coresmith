import React from 'react';

export default function GroupNode({ data, selected }) {
  return (
    <div className={`workflow-node group-node ${selected ? 'selected' : ''}`}>
      <div className="group-header">{data.label}</div>
      <div className="group-body">{data.children}</div>
    </div>
  );
}
