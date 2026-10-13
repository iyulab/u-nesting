# Algorithm Reference

U-Nesting implements multiple algorithms for spatial optimization. This guide explains each algorithm, its strengths, and when to use it.

The settings each section lists are the ones a request takes (`config` in the
[JSON schemas](../json-schema/README.md)); a key the engine does not read is
refused, not ignored. What a request cannot set is fixed, and named here.

## Overview

| Strategy | 2D | 3D | Speed | Quality | Best For |
|----------|:--:|:--:|:-----:|:-------:|----------|
| `blf` | ✓ | ✓ | ★★★★★ | ★★☆☆☆ | Quick estimates, simple shapes |
| `nfp` | ✓ | - | ★★★☆☆ | ★★★★☆ | Complex polygons, high quality |
| `ep` | - | ✓ | ★★★★☆ | ★★★★☆ | 3D packing, good default |
| `ga` | ✓ | ✓ | ★★☆☆☆ | ★★★★☆ | Optimization, diverse solutions |
| `brkga` | ✓ | ✓ | ★★☆☆☆ | ★★★★★ | Best quality, production use |
| `sa` | ✓ | ✓ | ★★☆☆☆ | ★★★★☆ | Global optimization |
| `gdrr` | ✓ | - | ★★★☆☆ | ★★★★☆ | Hybrid, balanced |
| `alns` | ✓ | - | ★★☆☆☆ | ★★★★★ | Large instances |
| `milpexact` | ✓ | - | ★☆☆☆☆ | ★★★★★ | Optimal solutions (small instances; `milp` builds) |

## Heuristic Algorithms

### Bottom-Left Fill (BLF)

**Strategy key**: `blf`

The simplest and fastest placement algorithm. Places items at the bottom-left-most valid position.

**Algorithm**:
1. Sort items by decreasing area/volume
2. For each item, find the lowest available Y position
3. At that Y level, find the leftmost valid X position
4. Place the item and update available space

**Pros**:
- Extremely fast (O(n²) worst case)
- Predictable results
- Good for rectangular shapes

**Cons**:
- Poor quality for irregular shapes
- No optimization of placement order

**When to use**:
- Quick feasibility checks
- Rectangular or simple shapes
- Time-critical applications

```python
result = u_nesting.solve_2d(parts, sheet, config={"strategy": "blf"})
```

### No-Fit Polygon (NFP)

**Strategy key**: `nfp`

Uses No-Fit Polygon computation for accurate collision-free placement of irregular polygons.

**Algorithm**:
1. Compute NFP between all part pairs
2. For each placement position, use NFP to check validity
3. Select position that minimizes waste

**Pros**:
- Handles complex concave polygons
- Accurate collision detection
- Good quality results

**Cons**:
- Slower than BLF
- NFP computation can be expensive for complex shapes

**When to use**:
- Irregular polygon shapes
- When accuracy is more important than speed

```python
result = u_nesting.solve_2d(parts, sheet, config={"strategy": "nfp"})
```

### Extreme Point (EP)

**Strategy key**: `ep`

3D packing heuristic that tracks "extreme points" - corners of placed boxes where new items can be placed.

**Algorithm**:
1. Initialize with container corners as extreme points
2. For each item, evaluate all extreme points
3. Select the point that minimizes wasted space
4. Place item and update extreme points

**Pros**:
- Fast and efficient for 3D
- Good space utilization
- Handles many orientations

**Cons**:
- 3D only
- Greedy, no global optimization

**When to use**:
- 3D bin packing (default choice)
- Real-time packing applications

```python
result = u_nesting.solve_3d(boxes, container, config={"strategy": "ep"})
```

## Metaheuristic Algorithms

### Genetic Algorithm (GA)

**Strategy key**: `ga`

Classic genetic algorithm that evolves a population of placement sequences.

**Chromosome**: Permutation of item indices + rotation choices

**Algorithm**:
1. Initialize random population
2. Evaluate fitness (utilization, waste)
3. Select parents via tournament
4. Apply crossover (PMX for permutation)
5. Apply mutation (swap, rotate)
6. Repeat for N generations

**Parameters**:
- `population_size`: Number of individuals (default: 30 in 2D, 100 in 3D)
- `max_generations`: Number of generations (default: 50 in 2D, 500 in 3D)
- `crossover_rate`: Probability of crossover (default: 0.85)
- `mutation_rate`: Probability of mutation (default: 0.05)

A `population_size` or `max_generations` you set is the one the search runs
with; the defaults apply only when you leave them out.

**Pros**:
- Good exploration of solution space
- Handles complex constraints
- Parallel population evaluation (the `parallel` feature)

**Cons**:
- Slower than heuristics
- Parameter tuning needed

```python
config = {
    "strategy": "ga",
    "population_size": 100,
    "max_generations": 200
}
result = u_nesting.solve_2d(parts, sheet, config=config)
```

### Biased Random-Key GA (BRKGA)

**Strategy key**: `brkga`

Advanced genetic algorithm using random keys for better search characteristics.

**Chromosome**: Vector of random keys [0, 1] that decode to placement sequence

**Algorithm**:
1. Initialize population with random keys
2. Decode keys to placement sequence
3. Evaluate using placement heuristic
4. Partition population into elite/non-elite
5. Generate offspring: elite × elite, elite × non-elite, random immigrants
6. Repeat for N generations

**Parameters**:
- `population_size`: Total population (default: 30 in 2D, 50 in 3D)
- `max_generations`: Number of generations (default: 50 in 2D, 100 in 3D)

The elite fraction (0.2), the fraction of random immigrants (0.15) and the bias
toward the elite parent (0.7) are fixed.

**Pros**:
- Better convergence than standard GA
- Self-adaptive search intensity
- State-of-the-art quality

**Cons**:
- Computationally intensive
- Requires more parameter tuning

```python
config = {
    "strategy": "brkga",
    "population_size": 100,
    "max_generations": 300
}
result = u_nesting.solve_2d(parts, sheet, config=config)
```

### Simulated Annealing (SA)

**Strategy key**: `sa`

Single-solution metaheuristic inspired by metal annealing process.

**Algorithm**:
1. Start with initial solution
2. Generate neighbor by small modification
3. Accept if better, or probabilistically if worse
4. Reduce temperature over time
5. Return best solution found

**Parameters**: `time_limit_ms` and `seed`. The schedule is fixed: in 2D from
temperature 50 to 1, cooling rate 0.9, 20 moves per temperature, at most 500
iterations; in 3D from 100 to 0.1, cooling rate 0.95, 50 moves per temperature,
at most 10000 iterations.

**Pros**:
- Good at escaping local optima
- Simple to implement
- Memory efficient

**Cons**:
- Single solution focus
- Sensitive to cooling schedule

```python
config = {
    "strategy": "sa",
    "time_limit_ms": 10000,
    "seed": 7
}
result = u_nesting.solve_2d(parts, sheet, config=config)
```

### Goal-Driven Ruin and Recreate (GDRR)

**Strategy key**: `gdrr`

Hybrid algorithm that iteratively destroys and rebuilds parts of the solution.

**Algorithm**:
1. Generate initial solution
2. **Ruin**: Remove subset of placements based on goal (worst utilization, random, radial)
3. **Recreate**: Re-insert removed items using placement heuristic
4. Accept if improved
5. Repeat with adaptive ruin size

**Parameters**: `time_limit_ms` and `seed` (2D only). It runs at most 1000
iterations, ruining 10-30% of the layout each time.

**Pros**:
- Good balance of exploration/exploitation
- Adaptive search intensity
- Handles large instances well

**Cons**:
- More complex parameter space
- May converge prematurely

```python
config = {
    "strategy": "gdrr",
    "time_limit_ms": 10000
}
result = u_nesting.solve_2d(parts, sheet, config=config)
```

### Adaptive Large Neighborhood Search (ALNS)

**Strategy key**: `alns`

Advanced metaheuristic with multiple destroy/repair operators selected adaptively.

**Destroy Operators**:
- Random removal
- Worst removal (by utilization contribution)
- Related removal (nearby items)
- Radial removal (from random center)

**Repair Operators**:
- Greedy insertion (best position)
- Regret insertion (maximize opportunity cost)
- Random insertion

**Algorithm**:
1. Initialize with constructive heuristic
2. Select destroy operator (roulette wheel based on performance)
3. Select repair operator (roulette wheel)
4. Apply destroy then repair
5. Accept/reject based on simulated annealing criterion
6. Update operator weights based on success

**Parameters**: `time_limit_ms` and `seed` (2D only). It runs at most 1000
iterations, updating the operator weights every 50 with a reaction factor of
0.15.

**Pros**:
- State-of-the-art for large instances
- Self-tuning operator selection
- Robust across problem types

**Cons**:
- Complex implementation
- Longer runtime

```python
config = {
    "strategy": "alns",
    "time_limit_ms": 60000
}
result = u_nesting.solve_2d(parts, sheet, config=config)
```

## Exact Algorithms

### Mixed Integer Linear Programming (MILP)

**Strategy key**: `milpexact` (also `exact`; `hybridexact` tries it first and falls back to a search). Only a Rust build with the `milp` feature of `u-nesting-d2` has them; the published C, WebAssembly, Python and .NET libraries refuse the names (`invalid_option`).

Exact optimization using mathematical programming. Guarantees optimal solution but limited to small instances.

**Formulation**:
- Binary variables for item placement
- Continuous variables for positions
- Non-overlap constraints
- Boundary constraints

**Parameters**:
- `time_limit_ms`: Solver time limit (the optimality gap, 1%, is not a request setting)

**Pros**:
- Provably optimal solutions
- Clear optimality gap information

**Cons**:
- Only feasible for small instances (< 20 items)
- Exponential worst-case complexity

```rust
use u_nesting::{Config, Strategy};

let config = Config::new()
    .with_strategy(Strategy::MilpExact)
    .with_time_limit(120_000);
```

## Algorithm Selection Guide

### By Problem Size

| Items | Recommended Strategy |
|-------|---------------------|
| 1-10 | `milpexact` (optimal, `milp` builds) or `brkga` |
| 10-50 | `brkga` or `alns` |
| 50-200 | `alns` or `gdrr` |
| 200+ | `alns` with time limit |

### By Time Constraint

| Available Time | Strategy |
|----------------|----------|
| < 1 second | `blf` or `ep` |
| 1-10 seconds | `nfp` (2D) or `ga` |
| 10-60 seconds | `brkga` |
| > 60 seconds | `alns` |

### By Shape Complexity

| Shape Type | 2D Strategy | 3D Strategy |
|------------|-------------|-------------|
| Rectangles | `blf` | `ep` |
| Simple polygons | `nfp` | `ep` |
| Complex concave | `nfp` + `brkga` | `ep` + `brkga` |
| With holes | `nfp` + `brkga` | N/A |
