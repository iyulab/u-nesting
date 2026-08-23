# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-08-23

### Removed

- **`ExactConfig::use_symmetry_breaking` and `ExactConfig::use_cuts`** (and their
  `with_symmetry_breaking`/`with_cuts` builders) — both fields were silent no-ops.
  `use_symmetry_breaking` was read by the MILP solver but guarded a loop whose body
  never added any constraint; `use_cuts` was never read anywhere at all. Neither
  affected solution correctness (both were meant as pure performance hooks), but a
  documented config knob that does nothing is worse than no knob — a caller could
  reasonably believe toggling it changed solver behavior. Removed rather than
  implemented: no demonstrated performance need justifies the correctness risk of
  adding real symmetry-breaking constraints to a solver whose exactness this same
  release line has spent three prior fixes repairing (see 0.8.1 below). If a real
  need for either surfaces, they can be reintroduced with an actual implementation
  behind them. Breaking for any direct consumer of `u_nesting_core::exact::ExactConfig`
  building these fields; not reachable through the JSON/FFI/WASM config surface, which
  never exposed them.

## [0.8.1] - 2026-08-23

### Fixed

- **`Strategy::MilpExact` (and `HybridExact`'s exact-first attempt) placed
  almost nothing except axis-aligned rectangles.** After picking a candidate
  position, the solver converted its own grid-anchor coordinate directly into
  the piece's placement origin without translating by the piece's rotated
  local-frame offset — every other placement strategy already applies this
  conversion when turning a grid candidate into a `Placement`. For an
  axis-aligned rectangle that offset happens to be zero, so rectangles placed
  correctly by coincidence; any other shape (concave or convex, at any
  rotation the solver picked to minimize strip length) was silently
  positioned outside the boundary and then dropped, coming back as unplaced
  with no error. Fixed by applying the same anchor-to-origin correction the
  other strategies use.
- **`Strategy::MilpExact` could select two genuinely overlapping placements
  once a piece's own rotation was non-zero.** Its internal conflict check
  between two candidates compared their positions directly in global
  coordinates against a No-Fit Polygon computed assuming the reference piece
  sits unrotated at its own local origin — correct only while that reference
  candidate happened to be unrotated. Only reachable with two or more
  instances (a single piece never has a conflict to check), so this was
  latent for as long as the bug above kept multi-piece exact solves from
  reaching a rotated candidate at all. Fixed by carrying the No-Fit Polygon
  into absolute space the same way the other strategies already do: rotate
  it by the reference candidate's own rotation, then translate to its actual
  placement origin, before testing the other candidate's position against it.
- **`Strategy::MilpExact`'s `time_limit_ms` config bounded only its internal
  conflict-precomputation heuristic, never the actual MIP solve** — the one
  step that can legitimately run the longest on a hard instance. A caller
  setting a short time budget got no such bound where it mattered most.
  Fixed by passing the configured limit to the underlying HiGHS solver
  itself before solving.

## [0.8.0] - 2026-08-21

### Added

- **`allow_flip` (mirror reflection) is now supported end-to-end, across every
  placement strategy** (Bottom-Left Fill, NFP-guided BLF, Genetic Algorithm,
  BRKGA, Simulated Annealing, GDRR, ALNS, and the MILP-exact solver). A piece
  marked `allow_flip: true` can now be placed either in its original
  orientation or mirrored across its local axis, whichever a strategy finds
  better — previously `allow_flip: true` was rejected outright with an
  explicit error, since no strategy implemented mirroring yet. `Placement`'s
  existing `mirrored` field (and the 2D response's `flipped` field) now
  actually reflects what was solved; no wire-schema changes were needed since
  both fields already existed.

### Fixed

- **Mirrored placements could overlap an already-placed piece under NFP-guided
  strategies.** The boundary-clamp step that runs after a strategy picks a
  placement candidate computed the piece's bounding box from its *unmirrored*
  orientation regardless of whether the winning candidate was actually
  mirrored. For a piece whose local shape isn't symmetric about its own
  reference point, that bounding box is wrong for a mirrored placement — the
  clamp could then shift an already collision-free position into one that
  overlaps a previously placed piece. Fixed by computing the clamp bounds
  from the mirrored orientation whenever the candidate is mirrored. Only
  reachable together with the `allow_flip` support above, since mirroring had
  no live path before this release.
- **`Strategy::MilpExact` and `Strategy::HybridExact` never actually reached a
  working solver.** Both called into an older continuous-position MILP
  formulation that placed nothing for any input through the public solve
  entry point — not even a single unrotated, unmirrored rectangle — so
  `MilpExact` always returned every piece unplaced and `HybridExact` always
  silently fell back to its heuristic path. Fixed by routing both to the
  NFP Covering Model formulation that already had working, tested coverage
  (including the `allow_flip` support above); the older module is removed.

## [0.7.2] - 2026-07-15

Documentation-only clarification from continued fabric-cutting dogfooding. No
runtime behaviour changes; fully backward compatible.

### Documentation

- **`used_utilization` semantics clarified across all bindings (Rust, C#,
  Python).** The field is a *packing-density* metric: its denominator is the
  tight bounding box of the placed pieces, shrinking on **both** the width and
  length axes. For **fixed-width stock** (fabric rolls, coil, sheet stock) the
  boundary width is real consumed material, not padding, so collapsing the width
  axis over-reports how much material was saved (measured up to +13.4%p on a
  sleeve layout using 79% of a 1580 mm roll width). The docs now state this
  explicitly and direct fixed-width consumers to compute
  `placed_area / (boundary_width × used_bounding_box[1])` instead — the consumed
  length (`used_bounding_box[1]`) is already exposed and boundary-independent.
  A dedicated `stock_utilization` field is deferred pending cross-consumer
  demand (single-consumer use is already covered by the formula above).

## [0.7.1] - 2026-07-14

Follow-up hardening from the same fabric-cutting dogfooding pass. Behaviour of
the greedy strategies is unchanged; the fixes make the metaheuristic quality
floor and reproducibility actually hold. Fully backward compatible.

### Fixed

- **Metaheuristic quality floor now compares strip length, not bounding-box
  area.** `not_worse_than_blf` guaranteed only that GA/BRKGA/SA place at least as
  many pieces as BLF, comparing ties by bounding-box *area*. On an open-ended
  roll (fixed width, variable length) a taller, narrower layout has a smaller
  area yet consumes a *longer* strip, so a rotation-driven GA result that packed
  the same pieces into a longer roll than plain BLF slipped through the floor
  (e.g. 8 L-shapes: BLF 306, GA up to 812). The floor now compares the extent
  along the boundary's open (longer) axis — the material length consumers
  measure — so the stochastic strategies are provably never worse than BLF, and
  enlarging the allowed `rotations` set never worsens the result.
- **Seed reproducibility on the progress/callback path.** `solve_with_progress`
  (used by the FFI `solve_2d_with_callback` and the WASM/demo path) ran the GA
  from system entropy, so a seeded solve was non-deterministic whenever a
  progress callback was supplied. The seed is now threaded through the progress
  runner, matching the plain `solve` path.
- **Progress GA no longer runs the full default budget.** The callback path used
  the default 500 generations × 100 population instead of the capped 50 × 30 of
  the non-progress path, making a progress-driven solve far slower and prone to
  overrunning a modest time budget. Both paths now use the same caps.

### Added

- Robustness fuzz harness (`crates/d2/tests/fuzz_robustness.rs`): `solve` never
  panics on arbitrary/pathological polygons, and BLF/NFP layouts are verified
  overlap-free (convex and concave) with an independent `i_overlay` oracle.
- End-to-end tests that `guard_panic` converts an internal panic into a
  `success: false` response (previously verified only by construction).

## [0.7.0] - 2026-07-14

Dogfooding hardening pass from a fabric-cutting consumer (open-roll nesting of
irregular parts). All response additions are backward compatible — existing
fields keep their meaning and `#[serde(default)]` keeps old payloads valid.

### Added

- **Piece accounting on the solve response.** `SolveResponse`/`Pack3DResponse`
  gain `all_placed: bool` and `unplaced_count: usize`. `success` keeps its
  meaning ("solve completed without crashing"); full placement is now signalled
  by `all_placed` (`placements.len() == total_requested`), so an over-committed
  run no longer looks successful. `unplaced_count` is the instance-level count,
  complementing the deduplicated `unplaced` geometry-id list.
- **Padding-independent utilization.** 2D responses gain
  `used_bounding_box: [width, height]` (AABB of the placed pieces' actual
  footprint) and `used_utilization` (`piece_area / (used_w × used_h)`). Unlike
  `utilization`, these do not shrink as the boundary height grows — for an
  open-ended roll the larger axis is the material length consumed. Single-sheet
  solves only; `[0, 0]`/`0` for multi-sheet (per-sheet local frames make one
  footprint ill-defined) and on error.
- **Reproducible stochastic runs.** `Config` gains an optional `seed` (`seed?`
  in the binding config JSON), threaded into GA/BRKGA/SA so a fixed seed yields
  a deterministic layout. `None` seeds from system entropy.
- **`Strategy::parse`** — single source of truth for strategy-name parsing
  (case-insensitive, all aliases) shared by the C-FFI, Python, and WASM
  bindings.

### Fixed

- **Pieces no longer escape non-rectangular boundaries.** Placement containment
  used only an AABB test, so pieces could stick out past a triangular/concave
  boundary's edges and still be reported as placed. Containment now does real
  polygon-in-polygon testing (point-in-polygon + edge-intersection) for polygon
  boundaries, while plain rectangles and infinite strips keep the exact fast
  AABB path. Escaped pieces are rejected and correctly counted as unplaced.
- **Unknown strategy names are now an error** instead of a silent fall-back to
  `BottomLeftFill`, which hid consumer typos. (The FFI parser had also been
  missing `brkga`/`gdrr`/`alns`/`exact`, silently downgrading them to BLF.)
- **Invalid input geometry is rejected.** `Geometry2D::validate()` now rejects
  zero-area/collinear polygons (relative-epsilon signed-area test) and
  self-intersecting rings, and `allow_flip = true` errors until mirroring is
  implemented (was silently ignored). Arbitrary polygon boundaries get the same
  degeneracy/self-intersection checks.
- **Input validation can no longer be bypassed.** Per-geometry `validate()` was
  scattered across the per-strategy entry points, so the progress/callback path
  skipped it. Validation is hoisted to the top of `solve`,
  `solve_with_progress`, and `solve_multi_strip`.
- **Progress path respects the BLF quality floor.** The `solve_with_progress`
  GA branch bypassed the `not_worse_than_blf` guard that the plain `solve` path
  uses, so a callback-driven run could return a worse layout than BLF.
- **An internal panic no longer aborts the host process.** `panic = "abort"`
  is removed from the release profile so unwinding reaches the `catch_unwind`
  at the C-FFI entry points, which convert a panic into an error response
  instead of `SIGABRT`-ing the C#/Python/WASM host.

## [0.6.0] - 2026-07-07

### Fixed

- **Cutting-path sequencing is now wall-clock bounded** (`optimize_cutting_path`,
  all bindings). The 2-opt improvement phase had no time limit — only an
  iteration cap — and each candidate move re-evaluated the whole tour, so
  legitimate inputs (e.g. hundreds of identical parts filling a sheet) ran for
  many seconds and blocked the calling thread; in the browser this froze the
  tab (effectively a DoS on normal use). `CuttingConfig` gains
  `time_limit_ms` (default `5000`, `0` = unlimited), plumbed through the WASM
  and C-FFI `cutting_config` JSON (`time_limit_ms?: number`). On timeout the
  best sequence found so far is returned — cut order is heuristic, so early
  termination never invalidates the result. Independently, per-move cost
  evaluation no longer does a linear `contours.iter().find()`, dropping a
  factor of `n` from every 2-opt pass. Applies to both the legacy
  (`pierce_candidates <= 1`) and GTSP (`pierce_candidates > 1`) paths.

## [0.5.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.5.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry** (`@iyulab/u-nesting`). The npm
  package previously shipped only the wasm-bindgen *bundler*-target output,
  whose static `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in
  non-ESM packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports` map
  (`node` → CJS with filesystem wasm loading, `default` → bundler ESM).
  `require()`, native ESM `import`, and CJS TS runners all work without
  loader hooks. A pre-publish smoke test (CJS `require` + ESM `import`) now
  guards this path in CI. Wire schema and Rust/C#/Python APIs unchanged;
  0.5.1 is a lockstep version bump across all bindings (npm re-publish
  requires a new version, and binding versions are pinned in lockstep).

## [0.5.0] - 2026-06-21

### Added

- **`total_requested` on solve responses** (`solve_2d`/`solve_3d`). Both
  `SolveResponse` and `Pack3DResponse` now carry `total_requested: usize` — the
  Σ of every geometry's `quantity` (instance-level request total). `placements`
  is instance-level while `unplaced` lists **deduplicated** unique geometry IDs,
  so previously the per-instance unplaced count could not be derived from the
  response (`unplaced.len()` under-reports when a multi-quantity geometry fails).
  Consumers can now compute it directly as `total_requested - placements.len()`.
  The field is additive and `#[serde(default)]` (absent legacy/`optimize_cutting_path`
  passthrough payloads deserialize to `0`); on error responses it is `0`.
  Mirrored across the FFI, WASM, and Python bindings; C# binding models updated
  in lockstep. Resolves the instance-count under-reporting reported by consumers.

- **`config.multi_sheet` for 2D nesting** (`solve_2d`). New optional config flag
  (`Option<bool>`, default `false`). When `true`, parts that do not fit on a
  single sheet spill onto **additional sheets** instead of becoming `unplaced`:
  `sheets_used` reports the sheet count and each placement's `sheet_index`
  selects its sheet. Placement coordinates are **sheet-local** (relative to each
  sheet's origin), so consumers can render per-sheet panels without recomputing
  offsets. Previously the multi-sheet solver (`solve_multi_strip`) existed but was
  reachable only from the benchmark harness — WASM/FFI/Python/C# now expose it via
  the flag. The field is additive and optional (absent → single-sheet solve).
  Resolves the schema-vs-capability gap where `sheets_used`/`sheet_index` were
  declared but never exceeded 1/0.

### Fixed

- `SolveSummary.total_requested` previously computed `placements.len() +
  unplaced.len()`, which **undercounted** whenever a multi-quantity geometry was
  partially/fully unplaced (because `unplaced` is deduplicated). It now uses the
  authoritative instance-level total recorded at solve time.

- **Multi-sheet overflow no longer silently loses instances.**
  `Nester2D::solve_multi_strip` reduced a geometry's *remaining* set by unique
  geometry ID, so once **any** instance of a `quantity > 1` geometry was placed,
  the rest were dropped from both `placements` **and** `unplaced` (silent loss).
  It now reduces remaining quantity by the **instance count** placed on each
  sheet, carrying the remainder forward, and assigns globally-unique
  `(geometry_id, instance)` pairs across sheets. Genuinely oversized geometries
  are reported via an after-loop `unplaced` sweep. `solve_multi_strip` also now
  records `total_requested` (it previously left it `0`).

- **C# config key drift corrected.** The C# `Config2D`/`Config3D` DTOs serialized
  the GA/BRKGA generation cap as `"generations"`, but the Rust wire contract
  (`ConfigRequest`, `deny_unknown_fields`) only accepts `"max_generations"` — so
  setting it from C# caused the FFI to **reject the whole request**. The property
  is renamed `Generations` → `MaxGenerations` (`[JsonPropertyName("max_generations")]`),
  and the same key was corrected across the user-guide examples. **Breaking for C#
  callers** that set `Config2D.Generations`. A new `UNesting.Tests` project plus a
  CI `csharp` job now pin every emitted C# config key to the canonical Rust wire
  names so this class of drift fails in CI.

### Security

- Upgrade PyO3 `0.24` → `0.29` to remediate two advisories flagged by
  `cargo audit`:
  - **RUSTSEC-2026-0176** — out-of-bounds read in `nth`/`nth_back` for `PyList`
    and `PyTuple` iterators (unchecked arithmetic). Fixed in PyO3 0.29.0.
  - **RUSTSEC-2026-0177** — missing `Sync` bound on `PyCFunction::new_closure`
    closures (thread-safety). Fixed in PyO3 0.29.0.
  Python binding return types (`solve_2d`/`solve_3d`) migrated from the
  prelude-removed `PyObject` to `Bound<'py, PyAny>`; no Python-facing API change.

## [0.4.0] - 2026-06-12

### Changed — BREAKING (wire schema)

- All request objects (`solve_2d`/`solve_3d`/`optimize_cutting_path` —
  including nested geometry, boundary, and config objects) now **reject
  unknown keys** with an explicit `unknown field` error instead of silently
  ignoring them (`serde(deny_unknown_fields)`). Typos and unsupported options
  previously failed silently; remove any extra keys when upgrading.
- Response wire types apply the same strictness when deserialized (relevant
  for the `optimize_cutting_path` `solve_result` passthrough).

### Added

- `Request2D`/`Request3D` gain an optional `mode` field (`"2d"`/`"3d"`) —
  the auto-detect `solve` entry point's discriminator is now a declared part
  of the schema instead of an incidentally-tolerated extra key. It is omitted
  from serialization when absent.

### Changed

- Dependency: `u-metaheur` `^0.2` → `^0.3`.

## [0.3.5] - 2026-06-04

### Fixed
- **3D Extreme-Point under-packing**: the EP heuristic precomputed a per-axis "residual"
  free-space approximation and gated candidate points on it. That model under-counted
  space for boxes placed flush against each other, discarded valid extreme points, and
  stalled the pack (perfect-fill scenarios placed only 4/8 cubes). Replaced with an exact
  fit test against the container bounds and placed boxes at placement time
  (`ExtremePointSet::fits_at`): perfect-fill now 8/8, EP places 18 boxes vs BLF's 13.
  Also corrected an `EP → BLF` mis-routing in `solve_with_progress`.

### Changed
- **3D gravity & stability now enforced in `solve`**: the physics/stability analyzer was
  built but never wired into the solver. Added `enforce_support`, which reuses the
  analyzer to drop unsupported boxes and recompute utilization, with the support floor at
  `floor_z = margin` (a hard-coded `floor_z = 0` removed every box when `margin > 0` and
  gravity was on). Gravity/stability toggles restored in the demo.

## [0.3.4] - 2026-06-04

### Fixed
- **WASM runtime failures (0.3.3 regression)**: solve panicked under `wasm32` because the
  SA loop was not gated for the single-threaded target and used a raw `Instant`. Promoted
  `core::timing::Timer` to a `web-time` backend so `time_limit` works in the browser, and
  gated parallel paths. A single-strip time budget could exceed the total limit; the
  `.max(5000)` floor now also takes `.min(total)`.
- **3D extreme-point out-of-bounds**: a swapped `fits` argument (`extreme_point.rs`) could
  place a box outside the container.
- **GA/BRKGA orientation mis-report**: `rotation_index` was left unset, so chosen
  orientations were reported incorrectly in the response.
- Added a WASM runtime smoke suite (`wasm_runtime_smoke.mjs`, 18 cases) to CI.

## [0.3.3] - 2026-06-03

### Fixed
- **3D FFI/WASM response schema**: 3D solve now emits a dedicated `Pack3DResponse`
  / `Placement3DResponse` wire format matching the C# `PackingResult` / `Placement3D`
  binding. Previously the 3D path reused the 2D `SolveResponse`, so C# (and npm)
  consumers silently lost each placement's depth (`z`), `orientation` label and
  `bin_index` (and received `sheets_used` instead of `bins_used`). Same class as the
  0.3.1→0.3.2 2D `rotation` array-vs-scalar fix.

### Added
- `Geometry3D::orientation_label(idx)` — maps an orientation index to an axis
  permutation string (`"xyz"`, `"xzy"`, …).
- `build_pack3d_response` (u-nesting-d3) — canonical 3D wire-response builder shared
  by the C FFI and WASM bindings.
- FFI JSON schema regression guards pinning the Rust output to the C# binding
  contract for both 2D and 3D (`schema_guard_2d/3d_matches_csharp_*`).

### ⚠️ BREAKING — 3D wire schema (migration from ≤ 0.3.2)
3D `solve` responses moved from the 2D `SolveResponse` shape to the dedicated
`Pack3DResponse` / `Placement3DResponse`. Consumers reading 3D output must rename:

| ≤ 0.3.2 (old) | 0.3.3+ (current) |
|---------------|------------------|
| `placement.geometry_id` | `placement.id` |
| `placement.position: [x, y, z]` | flat `placement.x`, `placement.y`, `placement.z` |
| `placement.rotation: [rx, ry, rz]` (numeric) | `placement.orientation: "xyz"` (axis-permutation **string**) |
| `placement.boundary_index` | `placement.bin_index` |
| `response.boundaries_used` | `response.bins_used` |
| `response.computation_time_ms` | `response.elapsed_ms` |

The numeric `rotation` → string `orientation` change is semantic: `orientation` is
an axis-permutation label (see `Geometry3D::orientation_label`), not Euler angles.
Map it to axis swaps in your renderer rather than treating it as a rotation vector.

## [0.3.2] - 2026-05-12

### ⚠️ BREAKING — 2D wire schema (`align …JSON schema with C# binding`)
2D `solve` `PlacementResponse` / `SolveResponse` JSON fields were renamed and
restructured to match the C# `Geometry2D` / nesting-result binding. This is a
silent runtime break for JSON consumers (the WASM API returns a JSON string, so
TypeScript/`tsc` does **not** catch it). Migration from 0.3.1:

| 0.3.1 (old) | 0.3.2+ (current) |
|-------------|------------------|
| `placement.geometry_id` | `placement.id` |
| `placement.position: [x, y]` | flat `placement.x`, `placement.y` |
| `placement.rotation: [r]` (array) | `placement.rotation` (scalar number) |
| `placement.boundary_index` | `placement.sheet_index` |
| `response.boundaries_used` | `response.sheets_used` |
| `response.computation_time_ms` | `response.elapsed_ms` |
| — | + `placement.flipped` (bool) |

### Fixed
- C# JSON schema mismatch: Rust 2D output now matches the C# binding contract.

### Added
- NuGet release workflow + `.targets` for native DLL auto-copy; auto-trigger on
  `.csproj` version change.

## [0.3.1] - 2026-03-09

### Fixed
- WASM-compatible `Timer` abstraction to prevent `Instant::now()` panic under
  `wasm32` (no monotonic clock in the browser target).

### Added
- npm publish workflow for `@iyulab/u-nesting`.

## [0.3.0] - 2026-03-08

### Added
- `u-nesting-wasm` crate — 2D nesting, 3D packing, and cutting-path optimization
  exposed via WebAssembly (JSON string in / JSON string out).
- Shared `api_types` module: request/response types extracted so the C FFI and
  WASM bindings share one JSON schema definition.
- Cutting-path optimization crate (GTSP solver, kerf compensation, lead-in/out,
  bridge/tab micro-joints, thermal HAZ model, common-edge detection).

### Changed
- `rayon` made optional across `core`/`d2`/`d3` for WASM compatibility; parallel
  vs sequential branching now selected via `cfg` attributes.

## [0.2.0] - 2026-02-09

### Changed

#### Dependencies
- **BREAKING**: Removed `geo`, `geo-types`, `parry2d`, `parry3d` dependencies
- Replaced `geo` polygon operations with `u-geometry` equivalents
- Replaced `parry2d`/`parry3d` collision detection (unused) — removed entirely
- Replaced direct `nalgebra` imports with `u-geometry` re-exports
- Updated `rand` 0.8 → 0.9 (135 API changes across 14 files)
- Updated `pyo3` 0.22 → 0.24
- MSRV raised from 1.75 to 1.82 (uses `is_none_or` stabilized in 1.82)

#### Code Quality
- Eliminated all `unwrap()` in library code (replaced with `expect()`)
- Fixed all clippy warnings across workspace
- Extracted shared placement utilities to reduce code duplication (-561 LOC)
- Replaced duplicate robust predicates with `u-geometry` re-exports

### Added
- `u-geometry` integration for polygon operations (area, centroid, perimeter, convex_hull)
- `u-geometry` integration for convex NFP/Minkowski sum computation
- Polygon with holes support via `u-geometry` (area_with_holes, centroid_with_holes)
- SA, BRKGA, ALNS, GDRR algorithm documentation
- 2D nesting benchmarks (criterion)
- Benchmark build check in CI

### Removed
- `geo` 0.29 dependency
- `geo-types` 0.7 dependency
- `parry2d` 0.17 dependency
- `parry3d` 0.17 dependency
- Duplicate geometry functions (~200 LOC replaced by u-geometry delegation)

### Dependencies
- `u-geometry` 0.1 — Computational geometry primitives
- `u-metaheur` 0.1 — Metaheuristic optimization
- `u-numflow` 0.1 — Mathematical primitives
- `i_overlay` 1.9 — Boolean polygon operations
- `rstar` 0.12 — R*-tree spatial indexing
- `rayon` 1.10 — Parallelization
- `rand` 0.9 — Random number generation
- `pyo3` 0.24 — Python bindings

## [0.1.0] - 2026-01-21

### Added

#### Core Library (`u-nesting-core`)
- **Solver Framework**: Generic `Solver` trait with `Config` and `Strategy` enum
- **Genetic Algorithm**: `GaRunner` with tournament selection, elitism, and parallel evaluation
- **BRKGA**: Biased Random-Key Genetic Algorithm with random-key encoding
- **Simulated Annealing**: Multiple cooling schedules (Geometric, Linear, Adaptive, Lundy-Mees)
- **Placement System**: `Placement` struct with position, rotation, and boundary tracking
- **Transform Utilities**: `Transform2D`, `Transform3D`, `AABB2D`, `AABB3D`
- **Memory Optimization**: `ObjectPool`, `ClearingPool`, `SharedGeometry`, `GeometryCache`, `ScratchBuffer`
- **Progress Callbacks**: `ProgressInfo`, `GaProgress`, `BrkgaProgress` for real-time feedback

#### 2D Nesting (`u-nesting-d2`)
- **Geometry2D**: Polygon representation with holes, area, centroid, convex hull
- **Boundary2D**: Rectangular and arbitrary polygon boundaries
- **Nester2D Solver**: Multiple placement strategies
  - Bottom-Left Fill (BLF)
  - NFP-guided placement
  - Genetic Algorithm optimization
  - BRKGA optimization
  - Simulated Annealing optimization
- **NFP Engine**: No-Fit Polygon computation
  - Convex polygons via Minkowski sum
  - Non-convex polygons via triangulation + union
  - Thread-safe caching system
  - Inner-Fit Polygon (IFP) with margin support
- **Spatial Index**: R*-tree based collision detection

#### 3D Bin Packing (`u-nesting-d3`)
- **Geometry3D**: Box representation with 6 orientation variants
- **Boundary3D**: Container with mass, gravity, and stability constraints
- **Packer3D Solver**: Multiple packing strategies
  - Layer-based packing
  - Extreme Point heuristic
  - Genetic Algorithm optimization
  - BRKGA optimization
  - Simulated Annealing optimization
- **Spatial Index**: AABB-based collision detection

#### FFI Layer (`u-nesting-ffi`)
- **C ABI**: `unesting_solve()`, `unesting_solve_2d()`, `unesting_solve_3d()`
- **JSON API**: Request/Response serialization with serde
- **API Versioning**: Version field in all responses (v1.0)
- **Error Codes**: `UNESTING_OK`, `UNESTING_ERR_NULL_PTR`, etc.
- **Header Generation**: cbindgen for C/C++ headers

#### Python Bindings (`u-nesting-python`)
- **PyO3 Integration**: Native Python module via maturin
- **Functions**: `solve_2d()`, `solve_3d()`, `version()`, `available_strategies()`
- **Type Stubs**: `.pyi` files for IDE autocompletion

#### Benchmark Suite (`u-nesting-benchmark`)
- **2D Benchmarks**: ESICUP dataset parser and runner
- **3D Benchmarks**: Martello-Pisinger-Vigo (MPV) instance generator
- **Result Analysis**: Strategy comparison, rankings, win matrices
- **Report Generation**: Markdown and JSON output formats

#### Documentation
- **JSON Schemas**: `request-2d.schema.json`, `request-3d.schema.json`, `response.schema.json`
- **API Documentation**: Module-level docs with usage examples
- **README**: Quick start guide with C# P/Invoke examples

### Performance
- Parallel NFP computation via rayon
- Parallel GA/BRKGA population evaluation
- Parallel SA restarts
- Thread-safe NFP caching

### Dependencies
- `geo` 0.29 - 2D geometry primitives
- `i_overlay` 1.9 - Boolean polygon operations
- `parry2d`/`parry3d` 0.17 - Collision detection
- `nalgebra` 0.33 - Linear algebra
- `rstar` 0.12 - R*-tree spatial indexing
- `rayon` 1.10 - Parallelization
- `pyo3` 0.22 - Python bindings

[Unreleased]: https://github.com/iyulab/U-Nesting/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/iyulab/U-Nesting/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/iyulab/U-Nesting/releases/tag/v0.1.0
