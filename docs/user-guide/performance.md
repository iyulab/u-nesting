# Performance Tuning Guide

This guide covers performance optimization for U-Nesting solvers.

## Benchmarks

Typical performance on modern hardware (AMD Ryzen 9 / Intel i9):

### 2D Nesting

| Items | Strategy | Time | Utilization |
|-------|----------|------|-------------|
| 10 | BLF | 5ms | 65-75% |
| 10 | BRKGA | 2s | 80-85% |
| 50 | BLF | 50ms | 60-70% |
| 50 | BRKGA | 10s | 75-82% |
| 100 | BLF | 200ms | 55-65% |
| 100 | ALNS | 30s | 72-80% |

### 3D Bin Packing

| Items | Strategy | Time | Utilization |
|-------|----------|------|-------------|
| 20 | EP | 10ms | 70-80% |
| 20 | BRKGA | 3s | 78-85% |
| 100 | EP | 100ms | 65-75% |
| 100 | BRKGA | 15s | 72-82% |
| 500 | EP | 1s | 60-70% |
| 500 | ALNS | 60s | 68-78% |

## Configuration Strategies

### Speed Priority

For real-time or interactive applications:

```python
config = {
    "strategy": "blf",  # or "ep" for 3D
    "time_limit_ms": 1000
}
```

### Quality Priority

For batch processing where quality matters:

```python
config = {
    "strategy": "brkga",
    "population_size": 100,
    "generations": 500,
    "time_limit_ms": 60000
}
```

### Balanced

For most production use cases:

```python
config = {
    "strategy": "brkga",
    "population_size": 50,
    "generations": 200,
    "time_limit_ms": 15000
}
```

## Algorithm-Specific Tuning

### BRKGA

```python
config = {
    "strategy": "brkga",

    # Population size: larger = better exploration, slower
    # Rule of thumb: 2-5x number of items
    "population_size": 100,

    # Generations: more = better quality, longer time
    "generations": 300,

    # Elite ratio: fraction kept as elites (0.1-0.3)
    "elite_ratio": 0.2,

    # Mutant ratio: random immigrants (0.05-0.2)
    "mutant_ratio": 0.1,

    # Bias: probability of inheriting from elite parent (0.5-0.8)
    "bias": 0.7
}
```

### Simulated Annealing

```python
config = {
    "strategy": "sa",

    # Initial temperature: higher = more exploration
    "initial_temp": 1000,

    # Cooling rate: slower cooling = better quality
    # Range: 0.9 (fast) to 0.999 (slow)
    "cooling_rate": 0.995,

    # Minimum temperature: stopping condition
    "min_temp": 0.01,

    # Iterations per temperature level
    "iterations_per_temp": 100
}
```

### ALNS

```python
config = {
    "strategy": "alns",

    # Segment size: iterations between weight updates
    "segment_size": 100,

    # Reaction factor: how quickly weights adapt (0.05-0.3)
    "reaction_factor": 0.1,

    # Destroy ratio range
    "min_destroy_ratio": 0.1,
    "max_destroy_ratio": 0.4,

    # Score rewards for operator selection
    "score_new_best": 33,
    "score_better": 9,
    "score_accepted": 3
}
```

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

# Original: 500 vertices
# Simplified: 50 vertices (10x speedup)
```

### Sorting Heuristics

Pre-sorting items can improve heuristic performance:

```python
# Sort by area (2D) or volume (3D) - decreasing
parts.sort(key=lambda p: -calculate_area(p['polygon']))

# Sort by "difficulty" (aspect ratio)
parts.sort(key=lambda p: -max_dimension(p) / min_dimension(p))
```

### Quantity Expansion

For items with high quantity, consider:

```python
# Instead of: {"id": "small", "quantity": 100}
# Use: 100 separate items for better placement diversity
items = [{"id": f"small_{i}", "polygon": small_poly, "quantity": 1}
         for i in range(100)]
```

## Memory Optimization

### Large Instance Handling

For 1000+ items:

```python
config = {
    "strategy": "alns",  # Lower memory than GA/BRKGA
    "time_limit_ms": 120000,

    # Limit population if using GA
    "population_size": 30
}
```

### Streaming Results

For very large outputs, process placements incrementally:

```c
// In callback
int process_progress(const char* json, void* data) {
    // Parse partial results from progress
    // Write to disk/stream instead of accumulating in memory
    write_partial_results(json);
    return 1;
}
```

## Parallelization

U-Nesting automatically parallelizes:

- **GA/BRKGA**: Population evaluation runs in parallel
- **NFP Computation**: Parallel NFP generation
- **ALNS**: Parallel neighborhood evaluation

### Controlling Parallelism

```bash
# Limit thread count (environment variable)
RAYON_NUM_THREADS=4 ./your_app

# Or in Rust code
rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build_global()
    .unwrap();
```

## Profiling

### Time Breakdown

Use progress callback to measure phases:

```python
import time

phase_times = {}
last_phase = None
phase_start = time.time()

def track_phases(progress):
    global last_phase, phase_start, phase_times

    current_phase = progress['phase']
    if current_phase != last_phase:
        if last_phase:
            phase_times[last_phase] = time.time() - phase_start
        phase_start = time.time()
        last_phase = current_phase
    return True

result = u_nesting.solve_2d_with_progress(parts, sheet, callback=track_phases)
print("Phase times:", phase_times)
```

### Memory Profiling

```bash
# Rust memory profiling
RUSTFLAGS="-C target-cpu=native" \
cargo build -p u-nesting-ffi --release

valgrind --tool=massif ./your_app
ms_print massif.out.*
```

## Best Practices Summary

1. **Start with heuristics** (BLF/EP) for quick baseline
2. **Use BRKGA** for production quality
3. **Set time limits** to prevent runaway computation
4. **Simplify polygons** when precision isn't critical
5. **Pre-sort items** by size for heuristic boost
6. **Monitor progress** to detect convergence
7. **Profile before optimizing** - find actual bottlenecks

## Troubleshooting Performance

### Slow NFP Computation

**Symptoms**: Long time in "Computing NFPs" phase

**Solutions**:
- Simplify polygon vertices
- Use BLF strategy instead
- Reduce rotation options

### Memory Exhaustion

**Symptoms**: Process killed, swap usage

**Solutions**:
- Reduce population size
- Use ALNS instead of BRKGA for large instances
- Process in batches

### Poor Convergence

**Symptoms**: Utilization plateaus early

**Solutions**:
- Increase population diversity (higher mutant ratio)
- Use slower cooling for SA
- Try different strategies (GA vs SA vs ALNS)
