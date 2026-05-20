/**
 * Mock ELK.js for Jest tests.
 * Returns nodes at their original positions since layout isn't relevant for
 * component rendering tests.
 */
class ELK {
  async layout(graph) {
    return {
      ...graph,
      children: (graph.children || []).map((child, i) => ({
        ...child,
        x: (child.x ?? 0) + i * 250,
        y: (child.y ?? 0) + i * 150,
        width: child.width || 200,
        height: child.height || 70,
        children: (child.children || []).map((grandchild, j) => ({
          ...grandchild,
          x: (grandchild.x ?? 0) + j * 200,
          y: (grandchild.y ?? 0) + j * 100,
          width: grandchild.width || 200,
          height: grandchild.height || 70,
        })),
      })),
    };
  }
}

module.exports = ELK;
module.exports.default = ELK;
