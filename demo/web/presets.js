// 데모 샘플 데이터. solve_2d / solve_3d 요청 JSON 형태와 동일.
export const presets2d = {
  rectangles: {
    label: '사각형 혼합',
    geometries: [
      { id: 'A', polygon: [[0, 0], [100, 0], [100, 50], [0, 50]], quantity: 4, rotations: [0, 90] },
      { id: 'B', polygon: [[0, 0], [60, 0], [60, 80], [0, 80]], quantity: 3, rotations: [0, 90] },
      { id: 'C', polygon: [[0, 0], [40, 0], [40, 40], [0, 40]], quantity: 5, rotations: [0] },
    ],
    boundary: { width: 400, height: 300 },
  },
  lshape: {
    label: 'L자 형상',
    geometries: [
      { id: 'L', polygon: [[0, 0], [80, 0], [80, 30], [30, 30], [30, 80], [0, 80]], quantity: 6, rotations: [0, 90, 180, 270] },
    ],
    boundary: { width: 300, height: 300 },
  },
};

export const presets3d = {
  boxes: {
    label: '박스 혼합',
    geometries: [
      { id: 'big', dimensions: [40, 40, 40], quantity: 4 },
      { id: 'mid', dimensions: [30, 20, 25], quantity: 6 },
      { id: 'small', dimensions: [15, 15, 30], quantity: 8 },
    ],
    boundary: { dimensions: [100, 100, 100] },
  },
};
