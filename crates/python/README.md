# U-Nesting

A high-performance 2D/3D spatial optimization engine for nesting and bin packing problems.

[![PyPI version](https://badge.fury.io/py/u-nesting.svg)](https://badge.fury.io/py/u-nesting)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **2D Nesting**: Optimal placement of irregular polygons on sheets
- **3D Bin Packing**: Efficient box placement in containers
- **Multiple Algorithms**: BLF, NFP-guided, Genetic Algorithm, BRKGA, Simulated Annealing
- **High Performance**: Written in Rust with Python bindings
- **Type Hints**: Full type annotation support

## Installation

```bash
pip install u-nesting
```

## Quick Start

### 2D Nesting

```python
import u_nesting

# Define polygons to nest
geometries = [
    {
        "id": "part1",
        "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
        "quantity": 5,
        "rotations": [0, 90, 180, 270]
    },
    {
        "id": "triangle",
        "polygon": [[0, 0], [80, 0], [40, 60]],
        "quantity": 3
    }
]

# Define sheet boundary
boundary = {"width": 500, "height": 300}

# Configure solver
config = {
    "strategy": "nfp",      # Options: blf, nfp, ga, brkga, sa
    "spacing": 2.0,         # Gap between parts
    "time_limit_ms": 30000  # 30 second timeout
}

# Solve
result = u_nesting.solve_2d(geometries, boundary, config)

print(f"Utilization: {result['utilization']:.1%}")
print(f"Placed: {len(result['placements'])} items")
for p in result['placements']:
    print(f"  {p['geometry_id']}[{p['instance']}]: ({p['position'][0]:.1f}, {p['position'][1]:.1f})")
```

### 3D Bin Packing

```python
import u_nesting

# Define boxes to pack
geometries = [
    {
        "id": "small",
        "dimensions": [20, 20, 20],
        "quantity": 10
    },
    {
        "id": "large",
        "dimensions": [40, 30, 25],
        "quantity": 5,
        "mass": 2.5  # Optional weight
    }
]

# Define container
boundary = {
    "dimensions": [200, 150, 100],
    "max_mass": 50.0,    # Optional mass limit
    "gravity": True,     # Stack from bottom
    "stability": True    # Require stable placement
}

# Configure solver
config = {
    "strategy": "ep",       # Extreme Point heuristic
    "time_limit_ms": 10000
}

# Solve
result = u_nesting.solve_3d(geometries, boundary, config)

print(f"Utilization: {result['utilization']:.1%}")
print(f"Containers used: {result['boundaries_used']}")
```

## API Reference

### `solve_2d(geometries, boundary, config=None) -> dict`

Solve a 2D nesting problem.

**Parameters:**
- `geometries`: List of geometry definitions
  - `id` (str): Unique identifier
  - `polygon` (list): Vertices as [[x, y], ...]
  - `quantity` (int): Number of copies (default: 1)
  - `rotations` (list): Allowed rotation angles in degrees
  - `allow_flip` (bool): Allow horizontal flip
  - `holes` (list): Interior holes as list of polygons
- `boundary`: Sheet definition
  - `width`, `height` (float): Rectangle dimensions, OR
  - `polygon` (list): Custom boundary shape
- `config`: Solver configuration (optional)
  - `strategy` (str): "blf", "nfp", "ga", "brkga", "sa"
  - `spacing` (float): Gap between geometries
  - `margin` (float): Gap from boundary
  - `time_limit_ms` (int): Timeout in milliseconds
  - `population_size` (int): GA/BRKGA population
  - `max_generations` (int): GA/BRKGA generations

**Returns:** Dictionary with:
- `success` (bool): Whether solve succeeded
- `placements` (list): Placement results
- `utilization` (float): Area utilization ratio
- `boundaries_used` (int): Number of sheets used
- `unplaced` (list): IDs of items that couldn't be placed
- `computation_time_ms` (int): Solve time

### `solve_3d(geometries, boundary, config=None) -> dict`

Solve a 3D bin packing problem.

**Parameters:**
- `geometries`: List of box definitions
  - `id` (str): Unique identifier
  - `dimensions` (list): [width, depth, height]
  - `quantity` (int): Number of copies
  - `mass` (float): Weight (optional)
- `boundary`: Container definition
  - `dimensions` (list): [width, depth, height]
  - `max_mass` (float): Weight limit (optional)
  - `gravity` (bool): Enable gravity constraint
  - `stability` (bool): Enable stability constraint
- `config`: Same as solve_2d, plus:
  - `strategy`: "blf", "ep", "ga", "brkga", "sa"

**Returns:** Same structure as solve_2d

## Strategy Selection Guide

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `blf` | Fast | Good | Large instances, quick results |
| `nfp` | Medium | Better | 2D with complex shapes |
| `ep` | Fast | Good | 3D bin packing |
| `ga` | Slow | Best | Small instances, max quality |
| `brkga` | Slow | Best | Complex constraints |
| `sa` | Medium | Better | Balanced speed/quality |

## Requirements

- Python 3.8+
- No additional dependencies

## License

MIT License - see [LICENSE](https://github.com/iyulab/U-Nesting/blob/main/LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/iyulab/U-Nesting)
- [Issue Tracker](https://github.com/iyulab/U-Nesting/issues)
- [Documentation](https://github.com/iyulab/U-Nesting/tree/main/docs)
