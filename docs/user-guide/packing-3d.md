# 3D Bin Packing Guide

This guide covers the 3D bin packing (box placement) functionality of U-Nesting.

## Overview

3D bin packing optimizes the placement of rectangular boxes in containers to minimize wasted space or number of bins used. Common applications include:

- Shipping container loading
- Warehouse pallet stacking
- Truck/van loading optimization
- Package arrangement

The request below is the JSON the C interface, the WebAssembly package and the
.NET package send; the Python package takes the same `geometries`, `boundary`
and `config` as dictionaries. Every key a request takes is listed here and in the
[JSON schemas](../json-schema/README.md). A key the engine does not read is
refused (`malformed_input`), never ignored.

```json
{
  "mode": "3d",
  "geometries": [
    {"id": "crate", "dimensions": [40, 30, 25], "quantity": 6, "mass": 12.0},
    {"id": "fridge", "dimensions": [60, 40, 90], "quantity": 2, "mass": 45.0,
     "orientation": "upright"}
  ],
  "boundary": {"dimensions": [200, 150, 120], "max_mass": 500,
               "gravity": true, "stability": true},
  "config": {"strategy": "ep", "time_limit_ms": 10000}
}
```

## Input Format

### Geometry Definition

Each box has:

<!-- fragment -->
```json
{
  "id": "box_001",
  "dimensions": [100, 50, 30],
  "quantity": 10,
  "mass": 2.5,
  "orientation": "upright"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `dimensions` | number[3] | Yes | [width, depth, height], each above 0; height is the vertical (z) |
| `quantity` | integer | No | Number of copies, at least 1 (default: 1) |
| `mass` | number | No | Mass of one copy, at least 0; counted against `max_mass` |
| `orientation` | string | No | `any` (default), `upright` or `fixed` |

The request has no other fields; an unknown one is refused (`malformed_input`).

### Orientations

`orientation` limits how a box may be turned:

| Value | Turns allowed |
|-------|---------------|
| `any` | All six axis-aligned orientations |
| `upright` | Height stays vertical; a quarter turn about it (width and depth swap) |
| `fixed` | None: width along x, depth along y, height along z |

Each placement reports how its box was turned as an axis permutation: which of the
box's own axes -- `x` its width, `y` its depth, `z` its height -- lies along the
container's x, y and z.

| Code | Container X | Container Y | Container Z |
|------|-------------|-------------|-------------|
| `xyz` | width | depth | height |
| `xzy` | width | height | depth |
| `yxz` | depth | width | height |
| `yzx` | depth | height | width |
| `zxy` | height | width | depth |
| `zyx` | height | depth | width |

`upright` produces only `xyz` and `yxz`.

### Boundary Definition

<!-- fragment -->
```json
{
  "dimensions": [500, 400, 300],
  "max_mass": 1000,
  "gravity": true,
  "stability": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dimensions` | number[3] | Required | Container [width, depth, height], each above 0 |
| `max_mass` | number | ∞ | Maximum total mass, above 0 |
| `gravity` | boolean | false | No box may float: each rests on the floor or on a box below |
| `stability` | boolean | false | Each box rests on a support under at least 70% of its base |

There is one container. A packing is checked against gravity and stability once
it is complete, and a box that breaks them is taken out and reported as unplaced.

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategy` | string | `"blf"` | `blf` (layer packing), `ep`, `ga`, `brkga` or `sa` |
| `spacing` | number | 0 | Minimum distance between boxes, at least 0 |
| `margin` | number | 0 | Offset from the container walls and floor, at least 0 |
| `time_limit_ms` | integer | 30000 | Maximum solving time, at least 0 (0 = no limit) |
| `target_utilization` | number | none | `ga`, `brkga`, `sa`: stop once every box is packed at this utilization or more, in [0, 1] |
| `population_size` | integer | 100 (`ga`), 50 (`brkga`) | GA/BRKGA population, at least 2; a value you set is run as given |
| `max_generations` | integer | 500 (`ga`), 100 (`brkga`) | GA/BRKGA generation budget, at least 0; a value you set is run as given |
| `crossover_rate` | number | 0.85 | GA crossover probability, in [0, 1] |
| `mutation_rate` | number | 0.05 | GA mutation probability, in [0, 1] |

A setting that only some strategies read (`target_utilization`, `population_size`, `max_generations`, `crossover_rate`, `mutation_rate`, `seed`) is refused as `invalid_option` when the request's strategy does not read it, with the strategies that do in the message -- it is not accepted and ignored.
| `seed` | integer | none | RNG seed of `ga`, `brkga`, `sa`: the same request and seed give the same packing when the run ends on its own budget, not on `time_limit_ms` |

`multi_sheet` is a 2D setting: a 3D request with it is refused (`invalid_option`,
`config.multi_sheet`). Gravity and stability belong to the `boundary`, not the
configuration.

## Output Format

The C interface and WebAssembly response:

<!-- fragment -->
```json
{
  "success": true,
  "placements": [
    {
      "id": "box_001",
      "bin_index": 0,
      "x": 0,
      "y": 0,
      "z": 0,
      "orientation": "xyz"
    }
  ],
  "bins_used": 1,
  "utilization": 0.78,
  "total_requested": 8,
  "unplaced": [],
  "unplaced_count": 0,
  "all_placed": true,
  "stop_reason": "completed",
  "elapsed_ms": 850
}
```

| Field | Description |
|-------|-------------|
| `success` | Whether the request was solved; `false` with `error`, `code` and `details` when it was refused |
| `placements` | List of box placements (instance-level) |
| `bins_used` | 1 when anything was packed, 0 otherwise |
| `utilization` | Volume utilization (0-1) |
| `total_requested` | Σ of every box's quantity (instance-level) |
| `unplaced` | **Deduplicated** IDs of boxes that couldn't be placed (not per-instance) |
| `unplaced_count` | Instance-level count of boxes that couldn't be placed |
| `all_placed` | Whether every requested instance was placed |
| `stop_reason` | Why the run stopped: `completed`, `target_reached`, `iteration_limit`, `no_improvement`, `time_limit` (the best packing found when `time_limit_ms` ran out) or `cancelled` |
| `elapsed_ms` | Solving time in milliseconds |

The Python result has the same values under its own names: `geometry_id`,
`position` (`[x, y, z]`) and `rotation` in each placement, `boundaries_used` and
`computation_time_ms` (see the [Python README](../../crates/python/README.md)).

### Placement Fields

| Field | Description |
|-------|-------------|
| `id` | Geometry ID |
| `instance` | Which copy of the box (0-based) |
| `bin_index` | Which container (0-based; always 0) |
| `x`, `y`, `z` | Position of box corner |
| `orientation` | Box orientation code |

## Examples

### Basic Packing

```python
import u_nesting

boxes = [
    {"id": "small", "dimensions": [20, 20, 20], "quantity": 50},
    {"id": "medium", "dimensions": [40, 30, 25], "quantity": 20},
    {"id": "large", "dimensions": [60, 40, 35], "quantity": 10}
]

container = {"dimensions": [200, 150, 120]}

result = u_nesting.solve_3d(boxes, container)

print(f"Volume utilization: {result['utilization']:.1%}")
print(f"Boxes that did not fit: {result['unplaced_count']}")
```

### With Physical Constraints

```python
boxes = [
    {
        "id": "heavy_base",
        "dimensions": [50, 30, 50],
        "quantity": 5,
        "mass": 20.0
    },
    {
        "id": "light_top",
        "dimensions": [40, 40, 30],
        "quantity": 10,
        "mass": 2.0
    },
    {
        "id": "cube",
        "dimensions": [30, 30, 30],
        "quantity": 20,
        "mass": 5.0
    }
]

container = {
    "dimensions": [200, 200, 150],
    "max_mass": 500,
    "gravity": True,
    "stability": True
}

result = u_nesting.solve_3d(
    boxes, container,
    config={"strategy": "ep"}
)
```

### Orientation Constraints

```python
# TV boxes that must stay upright
tv_box = {
    "id": "tv_55inch",
    "dimensions": [130, 20, 80],  # width x depth x height
    "quantity": 10,
    "orientation": "upright"  # Only turn about the vertical
}

# Cases that must keep their exact orientation
bottle_case = {
    "id": "wine_case",
    "dimensions": [40, 35, 30],
    "quantity": 20,
    "orientation": "fixed"  # No turn at all
}

result = u_nesting.solve_3d(
    [tv_box, bottle_case],
    {"dimensions": [250, 200, 180], "gravity": True}
)
```

### Progress Monitoring

Progress is reported through the C interface
(`unesting_solve_3d_with_progress`, see the [FFI guide](./ffi-integration.md))
and the .NET package (below). `blf` reports as it packs; the other strategies
run without reports.

### C# Example with Async

```csharp
using UNesting;
using UNesting.Models;

var packer = new Packer3D();

var request = new PackingRequest
{
    Geometries = new List<Geometry3D>
    {
        new Geometry3D
        {
            Id = "box_a",
            Width = 30, Depth = 20, Height = 15,
            Quantity = 100,
            Mass = 1.5
        }
    },
    Boundary = new Boundary3D
    {
        Width = 120, Depth = 100, Height = 80,
        MaxMass = 200,
        Gravity = true,
        Stability = true
    },
    Config = new Config3D
    {
        Strategy = Strategy3D.ExtremePoint,
        TimeLimitMs = 10000
    }
};

var progress = new Progress<ProgressInfo>(p =>
{
    Console.WriteLine($"Progress: {p.ItemsPlaced}/{p.TotalItems}");
});

var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
var result = await packer.SolveAsync(request, progress, cts.Token);

Console.WriteLine($"Placed {result.Placements.Count} boxes in {result.BinsUsed} bin(s)");
```

## Stability Analysis

When `stability: true`, a box is stable when the boxes or floor directly under
it support at least 70% of its base area:

```text
support_area / base_area >= 0.7
```

When only `gravity: true`, any support under a box is enough; a box with none
is floating. In both cases the finished packing is analyzed, the boxes that
fail are removed (they come back in `unplaced`), and the analysis repeats until
every remaining box is supported.

## Tips and Best Practices

### 1. Order by Size

For better results with heuristics:

```python
# Large items first often gives better packing
boxes.sort(key=lambda b: -np.prod(b['dimensions']))
```

### 2. Use Gravity for Realistic Packing

```python
# For truck loading, palletizing: constraints of the container
container = {
    "dimensions": [200, 150, 120],
    "gravity": True,
    "stability": True
}
result = u_nesting.solve_3d(boxes, container, config={"strategy": "ep"})
```

### 3. Handle Oversized Items

```python
result = u_nesting.solve_3d(boxes, container, config)

if result['unplaced']:
    oversized = [b for b in boxes if b['id'] in result['unplaced']]
    for box in oversized:
        dims = box['dimensions']
        cont = container['dimensions']
        print(f"{box['id']}: {dims} doesn't fit in {cont}")
```

### 4. One Container per Solve

A solve packs one container. To spread boxes over several, solve again with the
boxes the previous solve left unplaced.

### 5. Performance Tuning

For large instances (1000+ boxes), a search with a time limit:

```python
config = {
    "strategy": "brkga",
    "time_limit_ms": 60000
}
```

For real-time applications:

```python
config = {
    "strategy": "ep",  # Fast heuristic
    "time_limit_ms": 1000
}
```
