// Sample data for the demo. Same shape as solve_2d / solve_3d request JSON.
//
// Presets are tuned so the sheet is genuinely constrained: a naive strategy (BLF)
// leaves parts unplaced while metaheuristics (GA/SA/ALNS) pack more into the same
// sheet. This makes the strategy dropdown a real comparison rather than showing an
// identical "everything fits" result for every algorithm. See cycle-logs/cycle-01.md.
export const presets2d = {
  rectangles: {
    label: 'Rectangles',
    geometries: [
      { id: 'A', polygon: [[0, 0], [100, 0], [100, 50], [0, 50]], quantity: 5, rotations: [0, 90] },
      { id: 'B', polygon: [[0, 0], [60, 0], [60, 80], [0, 80]], quantity: 4, rotations: [0, 90] },
      { id: 'C', polygon: [[0, 0], [40, 0], [40, 40], [0, 40]], quantity: 6, rotations: [0, 90] },
    ],
    // 240x215 sheet is genuinely tight: both utilization and placed-count rank
    // BLF (67%/7) < NFP (73%/9) < GA/SA/ALNS (76%/12 of 15) — the two metrics agree.
    boundary: { width: 240, height: 215 },
  },
  lshape: {
    label: 'L-shape',
    geometries: [
      { id: 'L', polygon: [[0, 0], [80, 0], [80, 30], [30, 30], [30, 80], [0, 80]], quantity: 10, rotations: [0, 90, 180, 270] },
    ],
    // Concave parts rely on rotation to interlock: BLF ~4, metaheuristics ~6 of 10.
    boundary: { width: 210, height: 170 },
  },
  withHoles: {
    label: 'With holes (washer)',
    geometries: [
      {
        id: 'washer',
        polygon: [[0, 0], [80, 0], [80, 80], [0, 80]],
        holes: [[[25, 25], [55, 25], [55, 55], [25, 55]]],
        quantity: 9,
        rotations: [0, 90],
      },
    ],
    // Demonstrates parts-with-holes; tight sheet keeps utilization ~78%.
    boundary: { width: 250, height: 170 },
  },
  customBoundary: {
    label: 'Custom boundary (pentagon)',
    geometries: [
      { id: 'sq', polygon: [[0, 0], [40, 0], [40, 40], [0, 40]], quantity: 18, rotations: [0, 90] },
    ],
    // Non-rectangular boundary packed with 18 squares (~62%).
    boundary: { polygon: [[0, 0], [220, 0], [270, 140], [135, 230], [0, 140]] },
  },
};

export const presets3d = {
  boxes: {
    label: 'Mixed boxes',
    // mass lets the "Max mass" control constrain total placed weight (big=10, mid=5,
    // small=2). With no limit set it has no effect, so the default run is unchanged.
    geometries: [
      { id: 'big', dimensions: [40, 40, 40], quantity: 4, mass: 10 },
      { id: 'mid', dimensions: [30, 20, 25], quantity: 6, mass: 5 },
      { id: 'small', dimensions: [15, 15, 30], quantity: 8, mass: 2 },
    ],
    // 85x85x80 container is tight enough that strategies differentiate:
    // BLF ~13/63%, GA ~17/67%, SA fills all 18 (~69%). A 100^3 box fit everything
    // for every strategy (degenerate 40%). See cycle-logs/cycle-02.md.
    boundary: { dimensions: [85, 85, 80] },
  },
};
