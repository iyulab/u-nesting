# 3D Bin Packing Guide

This guide covers the 3D bin packing (box placement) functionality of U-Nesting.

## Overview

3D bin packing optimizes the placement of rectangular boxes in containers to minimize wasted space or number of bins used. Common applications include:

- Shipping container loading
- Warehouse pallet stacking
- Truck/van loading optimization
- Package arrangement

## Input Format

### Geometry Definition

Each box requires:

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
| `gravity` | boolean | false | Enable gravity simulation |
| `stability` | boolean | false | Check placement stability |

### Gravity and Stability

When `gravity: true`:
- Items fall to rest on surfaces below
- No floating placements allowed

When `stability: true`:
- Items must have sufficient support (configurable %)
- Center of mass must be over support area
- Stacking constraints enforced

## Configuration Options

```json
{
  "strategy": "ep",
  "time_limit_ms": 30000,
  "population_size": 50,
  "max_generations": 100
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategy` | string | "ep" | Algorithm strategy |
| `time_limit_ms` | integer | 30000 | Maximum solving time |
| `population_size` | integer | 100 (`ga`), 50 (`brkga`) | GA/BRKGA population, at least 2; a value you set is run as given |
| `max_generations` | integer | 500 (`ga`), 100 (`brkga`) | GA/BRKGA generation budget; a value you set is run as given |

## Output Format

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
  "bins_used": 2,
  "utilization": 0.78,
  "total_requested": 8,
  "unplaced": [],
  "elapsed_ms": 850
}
```

| Field | Description |
|-------|-------------|
| `success` | Whether solving completed |
| `placements` | List of box placements (instance-level) |
| `bins_used` | Number of bins/containers used |
| `utilization` | Volume utilization (0-1) |
| `total_requested` | Σ of every box's quantity (instance-level). Unplaced instances = `total_requested - placements.length` |
| `unplaced` | **Deduplicated** IDs of boxes that couldn't be placed (not per-instance) |
| `elapsed_ms` | Solving time in milliseconds |

### Placement Fields

| Field | Description |
|-------|-------------|
| `id` | Geometry ID |
| `bin_index` | Which container (0-based) |
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
print(f"Bins needed: {result['bins_used']}")
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
        "id": "stackable",
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

# Bottles that must be upright
bottle_case = {
    "id": "wine_case",
    "dimensions": [40, 35, 30],
    "quantity": 20,
    "orientation": "fixed"  # No rotation allowed
}

result = u_nesting.solve_3d(
    [tv_box, bottle_case],
    {"dimensions": [250, 200, 180], "gravity": True}
)
```

### Progress Monitoring

```python
def on_progress(progress):
    items_placed = progress['items_placed']
    total = progress['total_items']
    util = progress['utilization']
    print(f"Placed {items_placed}/{total} items, utilization: {util:.1%}")
    return True  # Continue

result = u_nesting.solve_3d_with_progress(
    boxes, container,
    callback=on_progress,
    config={"strategy": "brkga", "max_generations": 200}
)
```

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

When `stability: true`, the solver performs physics-based stability checks:

### Support Area Check

An item is stable if:

```
support_area / base_area >= support_ratio
```

Default `support_ratio` is 0.75 (75% support required).

### Center of Mass Check

The center of mass must project within the support polygon:

```
    ┌───────────┐
    │   CoM     │  ← Stable: CoM over support
    │    ↓      │
────┴───────────┴────
    ████████████
```

## Tips and Best Practices

### 1. Order by Size

For better results with heuristics:

```python
# Large items first often gives better packing
boxes.sort(key=lambda b: -np.prod(b['dimensions']))
```

### 2. Use Gravity for Realistic Packing

```python
# For truck loading, palletizing
config = {
    "strategy": "ep",
    "gravity": True,
    "stability": True
}
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

### 4. Multi-Bin Optimization

When using multiple bins, the solver automatically:
- Opens new bins as needed
- Tries to minimize total bins used
- Balances load across bins

### 5. Performance Tuning

For large instances (1000+ boxes):

```python
config = {
    "strategy": "alns",  # Best for large instances
    "time_limit_ms": 60000,
    "segment_size": 50
}
```

For real-time applications:

```python
config = {
    "strategy": "ep",  # Fast heuristic
    "time_limit_ms": 1000
}
```
