# Performance Tuning Guide

This guide covers what a request can tune, and what it cannot, for speed and
quality.

## Measuring

Solve times depend on the shapes, their count, the rotations allowed and the
machine, so measure on instances like yours. The benchmark runner in this
repository solves the ESICUP datasets with any set of strategies and reports
utilization and time:

```bash
cargo run --release -p u-nesting-benchmark --bin bench-runner -- list
cargo run --release -p u-nesting-benchmark --bin bench-runner -- run --dataset SHAPES
```

## Configuration Strategies

A request tunes a solve with the strategy, the time limit and -- for the
population-based searches -- the population and the generation budget. These
are the only tuning settings a request has (the full list is in the
[2D](./nesting-2d.md#configuration-options) and
[3D](./packing-3d.md#configuration-options) guides).

| Profile | `strategy` | `time_limit_ms` | `population_size` | `max_generations` |
|---------|------------|-----------------|-------------------|-------------------|
| Speed (interactive) | `blf`, `nfp` (2D) or `ep` (3D) | 1000 | -- | -- |
| Balanced | `brkga` | 15000 | 50 | 200 |
| Quality (batch) | `brkga` | 60000 | 100 | 500 |

A quality run as a request:

```json
{
  "geometries": [
    {"id": "bracket", "polygon": [[0, 0], [60, 0], [60, 20], [20, 20], [20, 60], [0, 60]],
     "quantity": 12, "rotations": [0, 90, 180, 270]}
  ],
  "boundary": {"width": 300, "height": 200},
  "config": {"strategy": "brkga", "population_size": 100, "max_generations": 500,
             "time_limit_ms": 60000, "seed": 7}
}
```

A search stops at whichever comes first: its generation budget, the time limit,
a layout reaching `target_utilization`, or no improvement for a while. With a
`seed`, a run that ends on its own budget (not on the clock) is reproduced
exactly.

## Algorithm-Specific Tuning

### GA

`population_size`, `max_generations`, `crossover_rate` (default 0.85) and
`mutation_rate` (default 0.05). The population is 30 and the budget 50
generations in 2D, 100 and 500 in 3D, unless set.

### BRKGA

`population_size` and `max_generations` (30 and 50 in 2D, 50 and 100 in 3D,
unless set). Rule of thumb: a population of 2-5 times the number of parts. The
elite fraction (0.2), the fraction of random immigrants (0.15) and the bias
toward the elite parent (0.7) are fixed; a request cannot change them.

### Simulated Annealing, GDRR and ALNS

These run with a fixed schedule; a request sets only their `time_limit_ms` and
`seed`. In 2D, SA runs from temperature 50 to 1 with a cooling rate of 0.9, 20
moves per temperature and at most 500 iterations; GDRR and ALNS run at most
1000 iterations each. In 3D, SA runs from 100 to 0.1 with a cooling rate of
0.95, 50 moves per temperature and at most 10000 iterations.

The Rust API runs each search with a configuration of the caller's own --
`u_nesting_d2::sa_nesting::run_sa_nesting` with an `SaConfig`, and the
equivalent functions in `gdrr_nesting`, `alns_nesting`, `brkga_nesting` and
`ga_nesting`.

### Time budget of a 2D search

A 2D search spends a quarter of `time_limit_ms` per strip, at least 5 s and
never more than the limit; with `time_limit_ms` 0 it keeps a budget of its own
(15 s for GA and BRKGA, 10 s for SA, GDRR and ALNS). It first runs the greedy
NFP placement inside that budget, then searches from it, and never returns a
layout worse than bottom-left fill.

## Input Optimization

### Polygon Simplification

Complex polygons slow down NFP computation:

```python
from shapely.geometry import Polygon

def simplify_polygon(vertices, tolerance=1.0):
    """Reduce vertex count while preserving shape."""
    poly = Polygon(vertices)
    simplified = poly.simplify(tolerance, preserve_topology=True)
    return list(simplified.exterior.coords)[:-1]  # Remove duplicate last point
```

### Rotations

Each allowed rotation is a placement candidate for every part: `[0, 90, 180,
270]` costs four times what `[0]` does. Allow the rotations the material allows,
no more.

## Parallelization

A Rust build with the `parallel` feature evaluates a GA or BRKGA population and
computes no-fit polygons on Rayon's thread pool. Without it -- and in the
published C, WebAssembly, Python and .NET libraries, which are built without it
-- a solve runs on the calling thread. Run several solves on several threads to
use more cores; each solve is independent.

### Controlling Parallelism

The parallel work runs on the pool the solve is called from:

```rust
use u_nesting::d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting::{Config, Solver};

let geometries = vec![Geometry2D::rectangle("plate", 40.0, 20.0).with_quantity(6)];
let boundary = Boundary2D::rectangle(200.0, 100.0);
let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
let result = pool.install(|| Nester2D::new(Config::default()).solve(&geometries, &boundary))?;
```

or for the whole process:

```bash
RAYON_NUM_THREADS=4 ./your_app
```

## Profiling

### Time Breakdown

Progress reports carry the phase and the elapsed time (C interface and .NET;
see the [FFI guide](./ffi-integration.md#progress-json-format)). `blf`, `nfp`
and `ga` report as they run.

### Memory Profiling

```bash
# Rust memory profiling
RUSTFLAGS="-C target-cpu=native" \
cargo build -p u-nesting-ffi --release

valgrind --tool=massif ./your_app
ms_print massif.out.*
```

## Best Practices Summary

1. **Start with heuristics** (BLF/NFP/EP) for a quick baseline
2. **Use a search** (GA/BRKGA) for production quality
3. **Set time limits** to prevent runaway computation
4. **Simplify polygons** when precision isn't critical
5. **Limit rotations** to what the material allows
6. **Seed** runs you need to reproduce

## Troubleshooting Performance

### Slow NFP Computation

**Symptoms**: A slow `nfp` solve, or a search whose first layout takes long

**Solutions**:
- Simplify polygon vertices
- Use BLF strategy instead
- Reduce rotation options

### Memory Exhaustion

**Symptoms**: Process killed, swap usage

**Solutions**:
- Reduce population size
- Process in batches

### Poor Convergence

**Symptoms**: Utilization plateaus early

**Solutions**:
- A larger population and generation budget
- A longer time limit
- Try different strategies (GA vs BRKGA vs ALNS)
