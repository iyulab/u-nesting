# @iyulab/u-nesting

**2D/3D Spatial Optimization Engine** — High-performance nesting and bin packing via WebAssembly.

[![npm](https://img.shields.io/npm/v/@iyulab/u-nesting.svg)](https://www.npmjs.com/package/@iyulab/u-nesting)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/iyulab/U-Nesting/blob/main/LICENSE-MIT)

## Installation

```bash
npm install @iyulab/u-nesting
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

A browser **without** a bundler is not supported: the package loads its `.wasm`
file with an ES module import, which browsers refuse (`application/wasm` is not a
module script type), so `<script type="module">` from a CDN fails, and CDN
re-bundling services fail on the same import. Use a bundler or Node.

## Functions

| Function | Description |
|----------|-------------|
| `solve_2d(json: string): string` | Solve a 2D nesting problem |
| `solve_3d(json: string): string` | Solve a 3D bin packing problem |
| `optimize_cutting_path(json: string): string` | Optimize cutting path for placed parts |
| `check_2d(json: string): string` | Check a 2D layout against its problem |
| `version(): string` | Get API version |
| `available_strategies(): string` | The strategies each dimension offers and the settings each reads |

All functions use **JSON string I/O**.

## Usage

### 2D Nesting

```javascript
import { solve_2d } from '@iyulab/u-nesting';

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
import { solve_3d } from '@iyulab/u-nesting';

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
import { solve_2d, optimize_cutting_path } from '@iyulab/u-nesting';

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
    interior_direction: "ccw",
    time_limit_ms: 2000            // Bound the sequencing pass (see below)
  }
})));
```

### `optimize_cutting_path` Request

```typescript
interface CuttingRequest {
  geometries: Geometry[];          // Same shape as solve_2d geometries
  solve_result: SolveResult;       // The parsed result returned by solve_2d
  cutting_config?: {
    kerf_width?: number;           // Kerf compensation width (default: 0 = off)
    pierce_weight?: number;        // Pierce-count weight in cost, >= 0 (default: 10)
    max_2opt_iterations?: number;  // 2-opt iteration cap (default: 1000)
    time_limit_ms?: number;        // Wall-clock budget (ms) for the 2-opt phase.
                                   //   0 = unlimited. Default: 5000.
    rapid_speed?: number;          // Units/s, > 0; for time estimation only (default: 1000)
    cut_speed?: number;            // Units/s, > 0; for time estimation only (default: 100)
    exterior_direction?: string;   // "ccw" | "cw" | "auto" (default: "auto")
    interior_direction?: string;   // "ccw" | "cw" | "auto" (default: "auto")
    home_position?: [number, number]; // Head start/end point (default: [0,0])
    pierce_candidates?: number;    // Candidate pierce points per contour (default: 1)
    tolerance?: number;            // Geometric comparison tolerance, >= 0 (default: 1e-6)
  };
}
```

> **`time_limit_ms` (anytime bound).** Cutting-path sequencing runs a
> nearest-neighbor construction followed by 2-opt improvement. The 2-opt
> neighborhood is `O(n²)` candidate moves per pass, so on large jobs (e.g. many
> identical parts filling a sheet) the iteration cap alone can run for seconds
> and block the calling thread — in the browser this freezes the whole tab.
> `time_limit_ms` caps that phase; when the budget is exceeded the best sequence
> found so far is returned. Cut order is a heuristic, so early termination never
> invalidates the result — it only trades some optimality for responsiveness.
> Set `0` to disable the wall-clock bound (native/batch callers only).

### `optimize_cutting_path` Response

```typescript
interface CuttingResponse {
  version: string;
  success: boolean;
  error: string | null;    // readable text; null on success
  code: string | null;     // stable reason, see "Refusals"; null on success
  details: RefusalDetails | null;
  sequence: {
    contour_id: number; geometry_id: string; instance: number;
    contour_type: "exterior" | "interior";
    pierce_point: [number, number]; cut_direction: "ccw" | "cw";
    rapid_from: [number, number] | null; rapid_distance: number; cut_distance: number;
  }[];
  total_cut_distance: number;
  total_rapid_distance: number;
  total_pierces: number;
  estimated_time_seconds: number;  // rapid distance / rapid_speed + cut distance / cut_speed
  efficiency: number;
  computation_time_ms: number;
  // Contours the plan cannot cut. A part listed here is NOT fully cut by `sequence`.
  skipped_contours: {
    contour_id: number; geometry_id: string; instance: number;
    contour_type: "exterior" | "interior";
    reason: "kerf_collapsed";          // smaller than the tool: half the kerf offset left nothing
    offset: number | null;             // the offset applied (+kerf/2 exterior, -kerf/2 interior)
  }[];
}
```

Check `skipped_contours` whenever `kerf_width` is set: a hole narrower than the
kerf has no tool path that keeps the part's dimensions, so it is left out of
`sequence` and reported here rather than dropped.

The kerf must fit between the parts: each cut takes a band `kerf_width` wide
beside its part, so a gap narrower than the kerf would cut one part into its
neighbour. Such a request is refused with `parameter_out_of_range` on
`cutting_config.kerf_width`, `max` set to the narrowest gap -- nest with a `spacing` of at least
the kerf width. Every contour of every placed part is either in `sequence` or in
`skipped_contours`.

## Input Schemas

### `solve_2d` Request

```typescript
interface Request2D {
  geometries: {
    id: string;
    polygon: [number, number][];       // Vertices (CCW)
    quantity?: number;                  // Default: 1
    allow_flip?: boolean;              // Default: false
    rotations?: number[];              // Allowed rotation angles in degrees (at least one;
                                       //   omit for a fixed orientation — [] is refused)
    holes?: [number, number][][];      // Interior holes
  }[];
  boundary: {
    width?: number;                    // Rectangle boundary
    height?: number;
    polygon?: [number, number][];      // Or custom polygon boundary
  };
  config?: {
    spacing?: number;                  // Minimum distance between parts (default: 0)
    margin?: number;                   // Minimum distance from a part to the boundary edge (default: 0)
    strategy?: string;                 // See available strategies
    population_size?: number;          // GA/BRKGA population, >= 2, run as given
                                       //   (omitted: 30, what the 2D searches are tuned for)
    max_generations?: number;          // GA/BRKGA generation budget, >= 0, run as given
                                       //   (omitted: 50)
    crossover_rate?: number;           // GA crossover rate (default: 0.85)
    mutation_rate?: number;            // GA mutation rate (default: 0.05)
    time_limit_ms?: number;            // Time limit in ms for the whole solve
    target_utilization?: number;       // Stop early if reached
    seed?: number;                     // RNG seed of ga, brkga, sa, gdrr, alns (see below)
    multi_sheet?: boolean;             // Spill overflow onto extra sheets (default: false)
  };
}
```

`solve_3d` takes the same `config` but `multi_sheet`, which it refuses (3D packs
one container; there are no sheets to spill onto). A `population_size`
or `max_generations` you set is the one the search runs with; omitted, each
search uses the budget it is tuned for:

| Strategy | `population_size` omitted | `max_generations` omitted |
|---|---|---|
| 2D `ga`, `brkga` | 30 | 50 |
| 3D `ga` | 100 | 500 |
| 3D `brkga` | 50 | 100 |

The other strategies have no population and do not read them. A run also ends on
`time_limit_ms` and when the best layout stops improving.

A search never returns a layout worse than a simple fill of the same request: it
is floored at that fill (fewer pieces placed, then a longer strip in 2D or a
lower utilization in 3D, loses to it).

`seed` makes the stochastic strategies reproducible: the same request with the
same `seed` returns the same layout -- **when the run ends on its own budget**
(generations, iterations, convergence), not on the clock. Where `time_limit_ms`
stops a search depends on the machine, so a run cut short by it is not
reproduced; in 2D a search also keeps a time budget of its own (15 s for `ga` and
`brkga`, 10 s for `sa`, `gdrr` and `alns`) when `time_limit_ms` is 0. Without a
`seed`, each run draws fresh randomness.

### Multi-Sheet Nesting

By default `solve_2d` packs into a **single** sheet; parts that do not fit become
`unplaced`. Set `config.multi_sheet: true` to spill overflow onto **additional
sheets**. Then:

- `sheets_used` reports how many sheets were needed.
- each placement's `sheet_index` selects its sheet.
- placement `x`/`y` are **sheet-local** — relative to that sheet's own origin —
  so you can render one panel per sheet without subtracting any offset.

```javascript
const res = JSON.parse(solve_2d(JSON.stringify({
  geometries: [{ id: "part", polygon: [[0,0],[100,0],[100,100],[0,100]], quantity: 20 }],
  boundary: { width: 300, height: 300 },
  config: { strategy: "blf", multi_sheet: true }
})));
// res.sheets_used === 3, res.placements.length === 20, res.unplaced === []
// group placements by sheet_index to draw each sheet
```

### Checking a layout

`check_2d` judges a 2D layout by the rules the nester places under: every piece
within the boundary and its `margin`, no two pieces on a sheet overlapping or
closer than `spacing`, each at a rotation its geometry allows and flipped only
where `allow_flip` is set, each copy placed once, and the geometries left short
listed in `unplaced`. It takes `{ problem, solve_result }` — the `solve_2d`
request unchanged, and a successful `solve_2d` response (the engine's own, or a
layout edited or produced elsewhere in the same shape). A layout that breaks
its problem is a result, not a refusal:

```javascript
import { solve_2d, check_2d } from '@iyulab/u-nesting';

const problem = {
  geometries: [{ id: "sq", polygon: [[0,0],[10,0],[10,10],[0,10]], quantity: 2 }],
  boundary: { width: 30, height: 20 },
  config: { strategy: "blf", spacing: 1 },
};
const solved = JSON.parse(solve_2d(JSON.stringify(problem)));
console.log(JSON.parse(check_2d(JSON.stringify({ problem, solve_result: solved }))).feasible); // true

// Move the second piece onto the first:
solved.placements[1].x = solved.placements[0].x;
solved.placements[1].y = solved.placements[0].y;
const check = JSON.parse(check_2d(JSON.stringify({ problem, solve_result: solved })));
console.log(check.violations); // [ { code: "overlap", placement: 0, other: 1 } ]
```

| `code` | Fields | Meaning |
|---|---|---|
| `outside_boundary` | `placement`, `id` | The piece leaves the boundary, or comes closer to its edge than `margin` |
| `overlap` | `placement`, `other` | Two pieces on one sheet share area (touching is not overlapping) |
| `too_close` | `placement`, `other`, `distance`, `spacing` | Two pieces on one sheet are closer than `spacing` (a piece in another's hole is measured to that hole's edge) |
| `rotation_not_allowed` | `placement`, `id`, `rotation` | A rotation (degrees) the geometry's `rotations` do not include |
| `flip_not_allowed` | `placement`, `id` | Flipped without `allow_flip` |
| `sheet_not_allowed` | `placement`, `sheet` | On a sheet after the first without `multi_sheet` |
| `unknown_geometry` | `placement`, `id` | A placement of an id no geometry has |
| `instance_out_of_range` | `placement`, `id`, `instance`, `quantity` | A copy index not below the geometry's `quantity` |
| `instance_repeated` | `placement`, `id`, `instance` | The same copy placed twice |
| `missing_from_unplaced` | `id` | Copies left unplaced, the geometry not in `unplaced` |
| `wrongly_unplaced` | `id` | In `unplaced` but every copy is placed, listed twice, or unknown |

A request `solve_2d` refuses is refused the same way (`success: false`, `code`),
and so is a `solve_result` that is itself a refusal.

### Which settings a strategy reads

Each strategy reads only some settings; a setting the chosen strategy does not read
is refused (`invalid_option`), not ignored. `available_strategies()` answers which,
from the same table the refusal is made from:

```javascript
import { available_strategies } from '@iyulab/u-nesting';

const catalog = JSON.parse(available_strategies());
// { "2d": [{ "name": "blf", "reads": ["spacing", "margin", "time_limit_ms", "multi_sheet"] },
//          { "name": "ga", "reads": ["spacing", "margin", "time_limit_ms", "target_utilization", ...] }, ...],
//   "3d": [{ "name": "blf", "reads": [...] }, ...] }
const reads = (dim, strategy, setting) =>
  catalog[dim].some((s) => s.name === strategy && s.reads.includes(setting));
console.log(reads('2d', 'nfp', 'target_utilization')); // false -- leave the field out
console.log(reads('2d', 'ga', 'target_utilization'));  // true
```

```typescript
type StrategyInfo = { name: string; reads: string[] };
type StrategyCatalog = { "2d": StrategyInfo[]; "3d": StrategyInfo[] };
```

### Available 2D Strategies

| Strategy | Description |
|----------|-------------|
| `blf` | Bottom-Left Fill (fast, deterministic) |
| `nfp` | NFP-Guided placement |
| `ga` | Genetic Algorithm |
| `brkga` | Biased Random-Key GA |
| `sa` | Simulated Annealing |
| `gdrr` | Goal-Driven Ruin and Recreate |
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

### `solve_2d` Response

```typescript
interface SolveResponse {
  version: string;
  success: boolean;
  error: string | null;    // readable text; null on success
  code: string | null;     // stable reason, see "Refusals"; null on success
  details: RefusalDetails | null;
  placements: {
    id: string;            // geometry id
    instance: number;      // 0-based copy index
    x: number;
    y: number;
    rotation: number;      // degrees
    sheet_index: number;   // 0-based sheet/bin index (see config.multi_sheet)
    flipped: boolean;      // mirrored
  }[];
  sheets_used: number;     // >1 only when config.multi_sheet is true
  utilization: number;
  total_requested: number; // Σ of every geometry's quantity (instance-level).
                           // unplaced instances = total_requested - placements.length
  unplaced: string[];      // deduplicated geometry ids (NOT per-instance), so
                           // unplaced.length under-reports the failed-instance count
  elapsed_ms: number;
}
```

### `solve_3d` Response

```typescript
interface Pack3DResponse {
  version: string;
  success: boolean;
  error: string | null;    // readable text; null on success
  code: string | null;     // stable reason, see "Refusals"; null on success
  details: RefusalDetails | null;
  placements: {
    id: string;            // geometry id
    instance: number;      // 0-based copy index
    bin_index: number;     // 0-based bin index
    x: number;
    y: number;
    z: number;             // depth/height position (min corner)
    orientation: string;   // axis permutation, e.g. "xyz", "xzy"
  }[];
  bins_used: number;
  utilization: number;
  total_requested: number; // Σ of every geometry's quantity (instance-level).
                           // unplaced instances = total_requested - placements.length
  unplaced: string[];      // deduplicated geometry ids (NOT per-instance), so
                           // unplaced.length under-reports the failed-instance count
  elapsed_ms: number;
}
```

### Refusals

The functions never throw: a request the engine cannot honour comes back with
`success: false`, a readable `error`, a stable `code` and the values behind it in
`details`, so a program can point at the input to change without parsing the text.

`parameter` is the path of the value in the request, whichever step refused it:
`config.spacing`, `cutting_config.kerf_width`, `boundary.width`,
`geometries[2].quantity`.

```typescript
interface RefusalDetails {
  parameter?: string;   // the path in the request: "config.spacing", "config.strategy", "geometries[2].orientation", ...
  id?: string;          // the geometry the refusal is about
  first?: number;       // duplicate_id: where the id appeared first ...
  index?: number;       // ... and where it appeared again
  min?: number; max?: number;  // closed bounds of the accepted range: the value may equal them
  exclusive_min?: number; exclusive_max?: number;  // open bounds: the value must lie strictly beyond them
  got?: number | string;       // the value given
  expected?: string[];         // the names an unknown option could have been
}
```

| `code` | `details` | Meaning |
|---|---|---|
| `parameter_out_of_range` | `parameter`, `got`, and per side `min` / `max` (closed) or `exclusive_min` / `exclusive_max` (open); an unbounded side has neither; `id` for a geometry's own value | A setting outside its range (`config.spacing` below 0, `config.target_utilization` outside [0, 1], `config.mutation_rate`, `cutting_config.kerf_width`, `cutting_config.cut_speed` / `rapid_speed` not above 0, `cutting_config.pierce_weight` / `tolerance` below 0, ...); a count below its least value, a negative one included (`geometries[i].quantity` below 1, `config.population_size` below 2, `cutting_config.pierce_candidates` below 1, `config.seed` or `config.time_limit_ms` below 0); a boundary measure not above 0 (`boundary.width`, `boundary.height`, `boundary.dimensions[i]`, `boundary.max_mass`, each with `exclusive_min: 0`); a box's `geometries[i].dimensions[j]` not above 0 or `geometries[i].mass` below 0 |
| `empty_input` | `parameter` | A list with nothing in it: `geometries` (nothing to place or pack) |
| `unknown_option` | `parameter`, `got`, `expected` | A name nobody knows: `config.strategy`, a geometry's `orientation`, a cut direction (`cutting_config.exterior_direction`, ...), the C ABI's `mode` |
| `invalid_option` | `parameter` | A setting that cannot be used here: a `config.strategy` this build or this dimension does not offer, `config.multi_sheet` in a 3D request (3D has no sheets), a failed `solve_result` given to cut |
| `invalid_geometry` | `id` (when known) | A polygon with fewer than 3 vertices, self-intersecting or degenerate; a placement naming no geometry |
| `duplicate_id` | `parameter`, `id`, `first`, `index` | Two geometries share an id |
| `invalid_boundary` | `parameter` (when one field is at fault) | A boundary of a shape the engine cannot place into: both or neither of size and polygon, a polygon with fewer than 3 vertices, of no area or crossing itself |
| `malformed_input` | `parameter` (the path of the field, e.g. `geometries[2].quantity`, when there is one) | Not JSON, or a missing, unknown or mistyped key -- a count given as a fraction (`2.5`) included |
| `cancelled` | — | The caller stopped the run (C ABI progress callback) |
| `internal` | — | A failure inside the engine, not caused by the request |

## Related

- [u-geometry](https://www.npmjs.com/package/@iyulab/u-geometry) — Computational geometry primitives
- [u-metaheur](https://crates.io/crates/u-metaheur) — Metaheuristic optimization framework
