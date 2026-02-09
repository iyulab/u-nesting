# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `u-numerics` 0.1 — Mathematical primitives
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
