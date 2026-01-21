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
  "fragile": false,
  "max_stack": 3,
  "orientations": ["xyz", "xzy", "yxz"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `dimensions` | number[3] | Yes | [width, height, depth] |
| `quantity` | integer | No | Number of copies (default: 1) |
| `mass` | number | No | Weight for stability calculations |
| `fragile` | boolean | No | If true, nothing stacks on top |
| `max_stack` | integer | No | Maximum items on top |
| `orientations` | string[] | No | Allowed orientations |

### Orientation Codes

Orientations specify how dimensions map to container axes:

| Code | Container X | Container Y | Container Z |
|------|-------------|-------------|-------------|
| `xyz` | width | height | depth |
| `xzy` | width | depth | height |
| `yxz` | height | width | depth |
| `yzx` | height | depth | width |
| `zxy` | depth | width | height |
| `zyx` | depth | height | width |

Default: All 6 orientations allowed.

For items that must stay upright (Y is vertical):

```json
{
  "orientations": ["xyz", "zxy"]  // Only rotations around Y axis
}
```

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
| `dimensions` | number[3] | Required | Container [W, H, D] |
| `max_mass` | number | ∞ | Maximum total mass |
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
  "support_ratio": 0.7,
  "population_size": 50,
  "generations": 100
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategy` | string | "ep" | Algorithm strategy |
| `time_limit_ms` | integer | 30000 | Maximum solving time |
| `support_ratio` | number | 0.75 | Min support area ratio for stability |
| `population_size` | integer | 50 | GA/BRKGA population |
| `generations` | integer | 100 | GA/BRKGA generations |

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
  "unplaced": [],
  "elapsed_ms": 850
}
```

| Field | Description |
|-------|-------------|
| `success` | Whether solving completed |
| `placements` | List of box placements |
| `bins_used` | Number of bins/containers used |
| `utilization` | Volume utilization (0-1) |
| `unplaced` | IDs of boxes that couldn't be placed |
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
        "mass": 20.0,
        "fragile": False
    },
    {
        "id": "fragile_top",
        "dimensions": [40, 40, 30],
        "quantity": 10,
        "mass": 2.0,
        "fragile": True  # Nothing can be placed on top
    },
    {
        "id": "stackable",
        "dimensions": [30, 30, 30],
        "quantity": 20,
        "mass": 5.0,
        "max_stack": 3  # Max 3 items on top
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
    config={"strategy": "ep", "support_ratio": 0.8}
)
```

### Orientation Constraints

```python
# TV boxes that must stay upright
tv_box = {
    "id": "tv_55inch",
    "dimensions": [130, 80, 20],  # W x H x D
    "quantity": 10,
    "orientations": ["xyz", "zxy"],  # Only rotate around Y
    "fragile": True
}

# Bottles that must be upright
bottle_case = {
    "id": "wine_case",
    "dimensions": [40, 35, 30],
    "quantity": 20,
    "orientations": ["xyz"]  # No rotation allowed
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
    config={"strategy": "brkga", "generations": 200}
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
            Dimensions = new[] { 30.0, 20.0, 15.0 },
            Quantity = 100,
            Mass = 1.5
        }
    },
    Boundary = new Boundary3D
    {
        Dimensions = new[] { 120.0, 100.0, 80.0 },
        MaxMass = 200,
        Gravity = true,
        Stability = true
    },
    Config = new Config3D
    {
        Strategy = "ep",
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

### Stacking Pressure

Items track cumulative mass above them:

```
total_mass_above <= max_stack * item_mass
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
