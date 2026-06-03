// Sample data for the demo. Same shape as solve_2d / solve_3d request JSON.
export const presets2d = {
  rectangles: {
    label: 'Rectangles',
    geometries: [
      { id: 'A', polygon: [[0, 0], [100, 0], [100, 50], [0, 50]], quantity: 4, rotations: [0, 90] },
      { id: 'B', polygon: [[0, 0], [60, 0], [60, 80], [0, 80]], quantity: 3, rotations: [0, 90] },
      { id: 'C', polygon: [[0, 0], [40, 0], [40, 40], [0, 40]], quantity: 5, rotations: [0] },
    ],
    boundary: { width: 400, height: 300 },
  },
  lshape: {
    label: 'L-shape',
    geometries: [
      { id: 'L', polygon: [[0, 0], [80, 0], [80, 30], [30, 30], [30, 80], [0, 80]], quantity: 6, rotations: [0, 90, 180, 270] },
    ],
    boundary: { width: 300, height: 300 },
  },
  withHoles: {
    label: 'With holes (washer)',
    geometries: [
      {
        id: 'washer',
        polygon: [[0, 0], [80, 0], [80, 80], [0, 80]],
        holes: [[[25, 25], [55, 25], [55, 55], [25, 55]]],
        quantity: 6,
        rotations: [0, 90],
      },
    ],
    boundary: { width: 320, height: 240 },
  },
  customBoundary: {
    label: 'Custom boundary (pentagon)',
    geometries: [
      { id: 'sq', polygon: [[0, 0], [40, 0], [40, 40], [0, 40]], quantity: 8, rotations: [0, 90] },
    ],
    boundary: { polygon: [[0, 0], [300, 0], [360, 180], [180, 300], [0, 180]] },
  },
};

export const presets3d = {
  boxes: {
    label: 'Mixed boxes',
    geometries: [
      { id: 'big', dimensions: [40, 40, 40], quantity: 4 },
      { id: 'mid', dimensions: [30, 20, 25], quantity: 6 },
      { id: 'small', dimensions: [15, 15, 30], quantity: 8 },
    ],
    boundary: { dimensions: [100, 100, 100] },
  },
};
