# @iyulab/u-nesting

**2D/3D Spatial Optimization Engine** — High-performance nesting and bin packing via WebAssembly.

[![npm](https://img.shields.io/npm/v/@iyulab/u-nesting.svg)](https://www.npmjs.com/package/@iyulab/u-nesting)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/iyulab/U-Nesting/blob/main/LICENSE-MIT)

## Installation

```bash
npm install @iyulab/u-nesting
```

## Functions

| Function | Description |
|----------|-------------|
| `solve_2d(json: string): string` | Solve a 2D nesting problem |
| `solve_3d(json: string): string` | Solve a 3D bin packing problem |
| `optimize_cutting_path(json: string): string` | Optimize cutting path for placed parts |
| `version(): string` | Get API version |
| `available_strategies(): string` | List available strategies |

All functions use **JSON string I/O**.

## Usage

### 2D Nesting

```javascript
import init, { solve_2d } from '@iyulab/u-nesting';

await init();

const result = JSON.parse(solve_2d(JSON.stringify({
  geometries: [
    {
      id: "part-A",
      polygon: [[0,0], [100,0], [100,50], [0,50]],
      quantity: 3,
      allow_flip: false,
      rotations: [0, 90, 180, 270]
    },
    {
      id: "part-B",
      polygon: [[0,0], [60,0], [60,80], [0,80]],
      quantity: 2
    }
  ],
  boundary: { width: 500, height: 300 },
  config: {
    spacing: 2.0,
    strategy: "ga",
    population_size: 50,
    max_generations: 100,
    time_limit_ms: 5000
  }
})));

console.log(result);
// { success: true, placements: [...], utilization: 0.85, boundaries_used: 1, ... }
```

### 3D Bin Packing

```javascript
import init, { solve_3d } from '@iyulab/u-nesting';

await init();

const result = JSON.parse(solve_3d(JSON.stringify({
  geometries: [
    { id: "box-1", dimensions: [10, 20, 15], quantity: 5 },
    { id: "box-2", dimensions: [8, 12, 10], quantity: 3, mass: 2.5 }
  ],
  boundary: {
    dimensions: [100, 100, 100],
    max_mass: 50.0,
    gravity: true,
    stability: true
  },
  config: {
    strategy: "ga",
    time_limit_ms: 3000
  }
})));
```

### Cutting Path Optimization

```javascript
import init, { solve_2d, optimize_cutting_path } from '@iyulab/u-nesting';

await init();

// First solve the nesting problem
const solveResult = JSON.parse(solve_2d(JSON.stringify({ /* ... */ })));

// Then optimize cutting path
const cutting = JSON.parse(optimize_cutting_path(JSON.stringify({
  geometries: [/* same geometries */],
  solve_result: solveResult,
  cutting_config: {
    kerf_width: 0.5,
    cut_speed: 100.0,
    rapid_speed: 500.0,
    exterior_direction: "cw",
    interior_direction: "ccw"
  }
})));
```

## Input Schemas

### `solve_2d` Request

```typescript
interface Request2D {
  geometries: {
    id: string;
    polygon: [number, number][];       // Vertices (CCW)
    quantity?: number;                  // Default: 1
    allow_flip?: boolean;              // Default: false
    rotations?: number[];              // Allowed rotation angles in degrees
    holes?: [number, number][][];      // Interior holes
  }[];
  boundary: {
    width?: number;                    // Rectangle boundary
    height?: number;
    polygon?: [number, number][];      // Or custom polygon boundary
  };
  config?: {
    spacing?: number;                  // Part spacing (default: 0)
    margin?: number;                   // Boundary margin (default: 0)
    strategy?: string;                 // See available strategies
    population_size?: number;          // GA/BRKGA population (default: 50)
    max_generations?: number;          // GA/BRKGA generations (default: 100)
    crossover_rate?: number;           // Crossover rate (default: 0.8)
    mutation_rate?: number;            // Mutation rate (default: 0.1)
    time_limit_ms?: number;            // Time limit in ms
    target_utilization?: number;       // Stop early if reached
  };
}
```

### Available 2D Strategies

| Strategy | Description |
|----------|-------------|
| `blf` | Bottom-Left Fill (fast, deterministic) |
| `nfp` | NFP-Guided placement |
| `ga` | Genetic Algorithm |
| `brkga` | Biased Random-Key GA |
| `sa` | Simulated Annealing |
| `gdrr` | Guided Destroy-Repair-Refine |
| `alns` | Adaptive Large Neighborhood Search |

### Available 3D Strategies

| Strategy | Description |
|----------|-------------|
| `blf` | Bottom-Left Fill |
| `ep` | Extreme Point |
| `ga` | Genetic Algorithm |
| `brkga` | Biased Random-Key GA |
| `sa` | Simulated Annealing |

## Response Schema

```typescript
interface SolveResponse {
  version: string;
  success: boolean;
  error?: string;
  placements: {
    geometry_id: string;
    instance: number;
    position: { x: number; y: number; z?: number };
    rotation: { angle?: number; rx?: number; ry?: number; rz?: number };
    boundary_index: number;
  }[];
  boundaries_used: number;
  utilization: number;
  unplaced: string[];
  computation_time_ms: number;
}
```

## Related

- [u-geometry](https://www.npmjs.com/package/@iyulab/u-geometry) — Computational geometry primitives
- [u-metaheur](https://crates.io/crates/u-metaheur) — Metaheuristic optimization framework
