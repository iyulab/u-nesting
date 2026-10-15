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
    "strategy": "nfp",      # Options: blf, nfp, ga, brkga, sa, gdrr, alns
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

> Nothing a request asks for is raised as an exception: every refusal comes
> back in the result dict, with `success` `False`, a stable `code` and the
> values behind it in `details` -- the same codes and paths the C and
> WebAssembly bindings give. Branch on `code`, not on the `error` text.
>
> - Keys are read strictly: an unrecognized key in `geometries`, `boundary` or
>   `config`, or a value of the wrong type -- a count given as a fraction
>   (`2.5`) included -- is `malformed_input` with `details` `parameter` the
>   path of the field (`config.stratgy`, `geometries[0].quantity`). A typo can
>   never silently fall back to a default.
> - A number that is not finite (`float("nan")`, `math.inf`) is
>   `value_not_finite`: `parameter` is the path of the value
>   (`config.spacing`), or of the list it is in with its position as `index`
>   (`geometries[0].polygon[2]`, `index` 1 for the y coordinate), and `id`
>   names the geometry. A value JSON cannot hold (a set, a non-string key) is
>   `malformed_input`.
> - A value outside its range (a negative one included) is
>   `parameter_out_of_range`: a geometry's `quantity` below 1 has `parameter`
>   `geometries[i].quantity` and the geometry's `id`; a `config` setting
>   (`population_size` below 2, `seed` below 0, ...) has `parameter`
>   `config.population_size`; a boundary side, dimension or `max_mass` of 0
>   or below names it (`boundary.width`, `boundary.dimensions[i]`,
>   `boundary.max_mass`, with `exclusive_min` 0).
> - An empty `geometries` list is `empty_input` (`parameter` `geometries`); a
>   strategy the dimension does not offer is `invalid_option` (`parameter`
>   `config.strategy`); a boundary given as both a rectangle and a polygon, or
>   half of a rectangle, is `invalid_boundary`.

- `geometries`: List of geometry definitions
  - `id` (str): Unique identifier
  - `polygon` (list): Vertices as [[x, y], ...]
  - `quantity` (int): Number of copies (default: 1)
  - `rotations` (list): Allowed rotation angles in degrees — at least one; omit it for a fixed orientation (an empty list is refused)
  - `allow_flip` (bool): Allow horizontal flip
  - `holes` (list): Interior holes as list of polygons
- `boundary`: Sheet definition
  - `width`, `height` (float): Rectangle dimensions, OR
  - `polygon` (list): Custom boundary shape
- `config`: Solver configuration (optional)
  - `strategy` (str): "blf", "nfp", "ga", "brkga", "sa", "gdrr", "alns"
  - `spacing` (float): Gap between geometries
  - `margin` (float): Gap from boundary
  - `time_limit_ms` (int): Timeout in milliseconds, at least 0 (0 = no
    limit; default 30000)
  - `target_utilization` (float): `ga`, `brkga`, `sa` stop once every part is
    placed at this utilization or more, in [0, 1]
  - `population_size` (int): GA/BRKGA population, at least 2, run as given.
    Omitted, each search uses the size it is tuned for: 30 in 2D
  - `max_generations` (int): GA/BRKGA generation budget, at least 0, run as
    given. Omitted, 50 in 2D
  - `max_iterations` (int): SA/GDRR/ALNS iteration budget, at least 1; a seeded
    run that ends on it rather than on the time limit is reproduced exactly
  - A setting only some strategies read (`target_utilization`, `population_size`,
    `max_generations`, `max_iterations`, `crossover_rate`, `mutation_rate`, `seed`) is refused as
    `invalid_option` under a strategy that does not read it, not ignored
  - `crossover_rate` / `mutation_rate` (float): GA crossover and mutation
    probabilities, in [0, 1] (default 0.85 / 0.05)
  - `seed` (int): RNG seed of `ga`, `brkga`, `sa`, `gdrr`, `alns` and `hybridexact`. The same
    problem with the same seed returns the same layout when the run ends on its
    own budget rather than on the clock (with `time_limit_ms` 0 a search runs to
    its own budget, `max_generations` / `max_iterations`)
  - `multi_sheet` (bool): Distribute overflow across multiple sheets (default:
    `False`). When `True`, parts that do not fit on one sheet spill onto extra
    sheets instead of becoming unplaced; `boundaries_used` reports the sheet count
    and each placement's `boundary_index` selects its sheet with sheet-local
    coordinates.

**Returns:** Dictionary with:
- `success` (bool): Whether solve succeeded
- `placements` (list): Placement results (instance-level)
- `utilization` (float): Area utilization ratio
- `boundaries_used` (int): Number of sheets used (>1 only when `multi_sheet=True`)
- `total_requested` (int): Σ of every geometry's quantity (instance-level total).
  Unplaced instance count = `total_requested - len(placements)`
- `unplaced` (list): **Deduplicated** IDs of items that couldn't be placed (not
  per-instance, so `len(unplaced)` under-reports the failed-instance count)
- `unplaced_count` (int): Instance-level count of unplaced instances
- `all_placed` (bool): Whether every requested instance was placed -- prefer it
  to `success`, which only says the solve was not refused
- `used_bounding_box` (list): `[width, height]` of the placed parts' footprint
  (2D; `[0, 0]` for 3D)
- `used_utilization` (float): Part area over `used_bounding_box` (2D; `0` for 3D)
- `stop_reason` (str): Why the run stopped — `completed` (the method ran to its own end), `optimal` (an exact method proved it), `target_reached` (every part placed at `target_utilization` or more), `iteration_limit`, `no_improvement` (the search stalled), `time_limit` (`time_limit_ms` ran out; the layout is the best found by then) or `cancelled`. `None` on a refusal
- `computation_time_ms` (int): Solve time
- `error` (str | None): Readable text when the request was refused
- `code` (str | None): Stable reason for that refusal (`invalid_geometry`,
  `duplicate_id`, `parameter_out_of_range`, ...)
- `details` (dict | None): The values behind `code` (`id`, `first`, `index`,
  `parameter`, `got`, ...). A range gives each bounded side as `min` / `max`
  when the bound itself is accepted, or `exclusive_min` / `exclusive_max` when
  it is not

### `solve_3d(geometries, boundary, config=None) -> dict`

Solve a 3D bin packing problem.

**Parameters:**
- `geometries`: List of box definitions
  - `id` (str): Unique identifier
  - `dimensions` (list): [width, depth, height]
  - `quantity` (int): Number of copies
  - `mass` (float): Weight, at least 0 (optional)
  - `orientation` (str): `"any"`, `"upright"` or `"fixed"` (optional)
- `boundary`: Container definition
  - `dimensions` (list): [width, depth, height], each above 0
  - `max_mass` (float): Weight limit, above 0 (optional)
  - `gravity` (bool): Enable gravity constraint
  - `stability` (bool): Enable stability constraint
- `config`: Same as solve_2d but `multi_sheet`, which is refused
  (`invalid_option`, `config.multi_sheet`: one container, no sheets), and:
  - `strategy`: "blf", "ep", "ga", "brkga", "sa"
  - `population_size` / `max_generations` omitted: 100 / 500 for `ga`, 50 /
    100 for `brkga`

**Returns:** Same structure as solve_2d

### `available_strategies() -> dict`

The strategies each dimension offers and the settings each reads:
`{"2d": [{"name": "blf", "reads": ["spacing", ...]}, ...], "3d": [...]}`. A setting the
chosen strategy does not read is refused (`invalid_option`), not ignored; this is the
table that refusal is made from, so a caller can offer only the settings that change
the run:

```python
import u_nesting

catalog = u_nesting.available_strategies()
reads = {s["name"]: set(s["reads"]) for s in catalog["2d"]}
assert "target_utilization" in reads["ga"]
assert "target_utilization" not in reads["nfp"]
```

### `codes() -> dict`

Every refusal `code` the functions answer with (with the `details` fields that can come
with it) and every violation `code` the checkers report:
`{"refusals": [{"code", "fields"}], "violations": [{"code", "fields", "checkers"}]}`. It is
read off the engine's own refusal and violation types, so a caller that translates codes
can check its list against the package's:

```python
import u_nesting

known = {r["code"] for r in u_nesting.codes()["refusals"]}
assert "parameter_out_of_range" in known
```

## Strategy Selection Guide

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `blf` | Fast | Good | Large instances, quick results |
| `nfp` | Medium | Better | 2D with complex shapes |
| `ep` | Fast | Good | 3D bin packing |
| `ga` | Slow | Best | Small instances, max quality |
| `brkga` | Slow | Best | Complex constraints |
| `sa` | Medium | Better | Balanced speed/quality |
| `gdrr` | Medium | Better | 2D, ruin-and-recreate on dense layouts |
| `alns` | Medium | Better | 2D, adaptive neighborhood search |

## Requirements

- Python 3.8+
- No additional dependencies

## License

MIT License - see [LICENSE](https://github.com/iyulab/U-Nesting/blob/main/LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/iyulab/U-Nesting)
- [Issue Tracker](https://github.com/iyulab/U-Nesting/issues)
- [Documentation](https://github.com/iyulab/U-Nesting/tree/main/docs)
