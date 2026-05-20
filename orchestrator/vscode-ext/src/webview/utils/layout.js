import ELK from 'elkjs/lib/elk.bundled.js';

const elk = new ELK();

const DEFAULT_NODE_WIDTH = 200;
const DEFAULT_NODE_HEIGHT = 70;
const BUS_NODE_WIDTH = 250;
const BUS_NODE_HEIGHT = 60;
const GROUP_NODE_PADDING = 40;

/**
 * Apply ELK layout to ReactFlow nodes and edges.
 *
 * Handles three node categories for architecture block diagrams:
 * 1. Regular block nodes (compute, memory, hwa, etc.)
 * 2. Bus hub nodes (node_type === 'bus') -- laid out as star-topology hubs
 * 3. Subsystem group nodes (is_subsystem === true) -- compound containers
 *
 * Virtual __start__/__end__ nodes are injected into the ELK graph so the
 * layered algorithm knows the true entry/exit points and breaks cycles
 * in the correct direction (back-edges go backward, not forward).
 */
export async function applyElkLayout(nodes, edges, direction = 'RIGHT') {
  const nodeIds = new Set(nodes.map((n) => n.id));

  // Keep edges that reference real nodes OR virtual __start__/__end__ nodes.
  const virtualIds = new Set(['__start__', '__end__']);
  const validEdges = edges.filter(
    (e) =>
      (nodeIds.has(e.source) || virtualIds.has(e.source)) &&
      (nodeIds.has(e.target) || virtualIds.has(e.target))
  );

  // Collect any virtual node IDs referenced by edges but missing from the node list
  const referencedVirtual = new Set();
  const entryNodeIds = new Set();
  const exitNodeIds = new Set();
  for (const e of validEdges) {
    if (virtualIds.has(e.source) && !nodeIds.has(e.source)) referencedVirtual.add(e.source);
    if (virtualIds.has(e.target) && !nodeIds.has(e.target)) referencedVirtual.add(e.target);
    if (e.source === '__start__' && nodeIds.has(e.target)) entryNodeIds.add(e.target);
    if (e.target === '__end__' && nodeIds.has(e.source)) exitNodeIds.add(e.source);
  }

  // Separate subsystem group nodes from child/leaf nodes
  const groupNodeIds = new Set();
  const childToParent = {};
  for (const node of nodes) {
    if (node.data?.is_subsystem) {
      groupNodeIds.add(node.id);
    }
    const parentId = node.data?.node_parentId;
    if (parentId && parentId !== 'graph_root' && nodeIds.has(parentId)) {
      childToParent[node.id] = parentId;
    }
  }

  // Build ELK nodes -- group nodes become compound containers
  const topLevelElkNodes = [];
  const childrenByGroup = {};

  for (const node of nodes) {
    const isBus = node.data?.node_type === 'bus';
    const isGroup = groupNodeIds.has(node.id);
    const parent = childToParent[node.id];

    const elkNode = {
      id: node.id,
      width: isBus ? BUS_NODE_WIDTH : (isGroup ? 0 : DEFAULT_NODE_WIDTH),
      height: isBus ? BUS_NODE_HEIGHT : (isGroup ? 0 : DEFAULT_NODE_HEIGHT),
    };

    if (!isGroup) {
      if (entryNodeIds.has(node.id)) {
        elkNode.layoutOptions = { ...elkNode.layoutOptions, 'elk.layered.layerConstraint': 'FIRST' };
      }
      if (exitNodeIds.has(node.id)) {
        elkNode.layoutOptions = { ...elkNode.layoutOptions, 'elk.layered.layerConstraint': 'LAST' };
      }
    }

    if (isGroup) {
      // Compound node: will contain children
      elkNode.layoutOptions = {
        'elk.algorithm': 'layered',
        'elk.direction': direction,
        'elk.padding': `[top=${GROUP_NODE_PADDING + 30},left=${GROUP_NODE_PADDING},bottom=${GROUP_NODE_PADDING},right=${GROUP_NODE_PADDING}]`,
        'elk.spacing.nodeNode': '40',
        'elk.layered.spacing.nodeNodeBetweenLayers': '60',
      };
      elkNode.children = [];
      elkNode.edges = [];
    }

    if (parent && !isGroup) {
      // This is a child of a group node
      if (!childrenByGroup[parent]) childrenByGroup[parent] = [];
      childrenByGroup[parent].push(elkNode);
    } else {
      topLevelElkNodes.push(elkNode);
    }
  }

  // Attach children to their group nodes
  for (const elkNode of topLevelElkNodes) {
    if (childrenByGroup[elkNode.id]) {
      elkNode.children = childrenByGroup[elkNode.id];
    }
  }

  // Add tiny virtual nodes with layer constraints
  for (const vid of referencedVirtual) {
    const opts = {};
    if (vid === '__start__') opts['elk.layered.layerConstraint'] = 'FIRST';
    if (vid === '__end__') opts['elk.layered.layerConstraint'] = 'LAST';
    topLevelElkNodes.push({
      id: vid,
      width: 1,
      height: 1,
      ...(Object.keys(opts).length > 0 ? { layoutOptions: opts } : {}),
    });
  }

  // Build ELK edges -- route intra-group edges inside their compound node
  const topLevelEdges = [];

  for (let i = 0; i < validEdges.length; i++) {
    const edge = validEdges[i];
    const elkEdge = {
      id: edge.id || `e${i}`,
      sources: [edge.source],
      targets: [edge.target],
    };

    // Check if both source and target belong to the same group
    const srcParent = childToParent[edge.source];
    const tgtParent = childToParent[edge.target];

    if (srcParent && srcParent === tgtParent) {
      // Intra-group edge: attach to the group node
      const groupElk = topLevelElkNodes.find(n => n.id === srcParent);
      if (groupElk && groupElk.edges) {
        groupElk.edges.push(elkEdge);
        continue;
      }
    }

    topLevelEdges.push(elkEdge);
  }

  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': direction,
      'org.eclipse.elk.direction': direction,
      'org.eclipse.elk.spacing.nodeNode': '60',
      'org.eclipse.elk.spacing.edgeNode': '40',
      'elk.spacing.nodeNode': '60',
      'elk.layered.spacing.nodeNodeBetweenLayers': '100',
      'elk.layered.spacing.edgeEdgeBetweenLayers': '30',
      'elk.layered.spacing.edgeNodeBetweenLayers': '40',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
      'elk.layered.thoroughness': '15',
      'elk.edgeRouting': 'SPLINES',
      'elk.spacing.componentComponent': '80',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
    },
    children: topLevelElkNodes,
    edges: topLevelEdges,
  };

  const layouted = await elk.layout(graph);

  // Flatten positioned nodes from the ELK result (including children of compounds)
  const positionMap = {};
  const sizeMap = {};

  function collectPositions(elkChildren, offsetX = 0, offsetY = 0) {
    for (const child of elkChildren || []) {
      positionMap[child.id] = { x: child.x + offsetX, y: child.y + offsetY };
      sizeMap[child.id] = { width: child.width, height: child.height };
      if (child.children) {
        collectPositions(child.children, child.x + offsetX, child.y + offsetY);
      }
    }
  }
  collectPositions(layouted.children);

  return nodes.map((node) => {
    const pos = positionMap[node.id];
    const size = sizeMap[node.id];
    if (!pos) return node;

    const isGroup = groupNodeIds.has(node.id);
    const result = {
      ...node,
      position: pos,
    };

    if (isGroup && size) {
      result.style = { ...node.style, width: size.width, height: size.height };
    } else if (size) {
      result.style = { ...node.style, width: size.width };
    }

    return result;
  });
}
