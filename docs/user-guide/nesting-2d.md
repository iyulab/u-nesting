# 2D Nesting Guide

This guide covers the 2D nesting (polygon placement) functionality of U-Nesting.

## Overview

2D nesting optimizes the placement of irregular polygons on rectangular sheets to minimize material waste. Common applications include:

- Sheet metal cutting
- Textile/leather cutting
- PCB panel arrangement
- Laser/waterjet cutting optimization

## Input Format

### Geometry Definition

Each geometry requires:

```json
{
  "id": "unique_identifier",
  "polygon": [[x1, y1], [x2, y2], ...],
  "quantity": 5,
  "rotations": [0, 90, 180, 270],
  "allow_flip": false,
  "holes": [[[hx1, hy1], [hx2, hy2], ...]]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `polygon` | number[][] | Yes | Outer contour vertices (CCW) |
| `quantity` | integer | No | Number of copies (default: 1) |
| `rotations` | number[] | No | Allowed angles in degrees |
| `allow_flip` | boolean | No | Allow mirror reflection |
| `holes` | number[][][] | No | Interior holes (CW orientation) |

### Polygon Orientation

- **Outer contour**: Counter-clockwise (CCW)
- **Holes**: Clockwise (CW)

```
     2
    /\
   /  \
  /    \
 /      \
0--------1   CCW for outer

  1------2
  |  **  |
  |  **  |   ** = hole (CW)
  0------3
```

### Boundary Definition

```json
{
  "width": 1000,
  "height": 500
}
```

For multiple sheets with different sizes:

```json
{
  "sheets": [
    {"width": 1000, "height": 500},
    {"width": 800, "height": 400}
  ]
}
```

## Configuration Options

```json
{
  "strategy": "brkga",
  "spacing": 2.0,
  "margin": 5.0,
  "time_limit_ms": 30000,
  "population_size": 100,
  "generations": 200,
  "rotation_step": 90
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategy` | string | "blf" | Algorithm strategy |
| `spacing` | number | 0 | Minimum distance between parts |
| `margin` | number | 0 | Distance from sheet edges |
| `time_limit_ms` | integer | 30000 | Maximum solving time |
| `population_size` | integer | 50 | GA/BRKGA population size |
| `generations` | integer | 100 | GA/BRKGA generations |
| `rotation_step` | number | 90 | Default rotation increment |

## Output Format

```json
{
  "success": true,
  "placements": [
    {
      "id": "part1",
      "sheet_index": 0,
      "x": 10.5,
      "y": 20.0,
      "rotation": 90,
      "flipped": false
    }
  ],
  "sheets_used": 2,
  "utilization": 0.82,
  "unplaced": ["large_part"],
  "elapsed_ms": 1500
}
```

| Field | Description |
|-------|-------------|
| `success` | Whether solving completed |
| `placements` | List of part placements |
| `sheets_used` | Number of sheets used |
| `utilization` | Material utilization (0-1) |
| `unplaced` | IDs of parts that couldn't be placed |
| `elapsed_ms` | Solving time in milliseconds |

### Placement Fields

| Field | Description |
|-------|-------------|
| `id` | Geometry ID |
| `sheet_index` | Which sheet (0-based) |
| `x`, `y` | Position of reference point |
| `rotation` | Rotation angle in degrees |
| `flipped` | Whether mirrored |

## Examples

### Basic Rectangle Nesting

```python
import u_nesting

parts = [
    {
        "id": f"rect_{i}",
        "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
        "quantity": 1
    }
    for i in range(10)
]

sheet = {"width": 500, "height": 300}

result = u_nesting.solve_2d(parts, sheet, config={"strategy": "blf"})
print(f"Placed {len(result['placements'])} parts")
print(f"Utilization: {result['utilization']:.1%}")
```

### Irregular Shapes with Rotation

```python
# L-shaped part
l_shape = {
    "id": "L_part",
    "polygon": [
        [0, 0], [60, 0], [60, 20],
        [20, 20], [20, 60], [0, 60]
    ],
    "quantity": 8,
    "rotations": [0, 90, 180, 270]
}

# T-shaped part
t_shape = {
    "id": "T_part",
    "polygon": [
        [0, 0], [60, 0], [60, 20],
        [40, 20], [40, 60], [20, 60],
        [20, 20], [0, 20]
    ],
    "quantity": 5,
    "rotations": [0, 90, 180, 270]
}

result = u_nesting.solve_2d(
    [l_shape, t_shape],
    {"width": 400, "height": 300},
    config={"strategy": "nfp", "spacing": 3.0}
)
```

### Parts with Holes

```python
# Washer/ring shape
washer = {
    "id": "washer",
    "polygon": [
        [0, 0], [50, 0], [50, 50], [0, 50]  # Outer square
    ],
    "holes": [
        [[15, 15], [35, 15], [35, 35], [15, 35]]  # Inner hole (CW)
    ],
    "quantity": 20
}

# Frame shape
frame = {
    "id": "frame",
    "polygon": [
        [0, 0], [100, 0], [100, 80], [0, 80]
    ],
    "holes": [
        [[10, 10], [90, 10], [90, 70], [10, 70]]
    ],
    "quantity": 5
}

result = u_nesting.solve_2d(
    [washer, frame],
    {"width": 500, "height": 400},
    config={"strategy": "nfp"}
)
```

### Progress Monitoring

```python
def progress_callback(progress):
    print(f"Iteration {progress['iteration']}: "
          f"utilization={progress['utilization']:.1%}")
    # Return False to cancel
    return True

result = u_nesting.solve_2d_with_progress(
    parts, sheet,
    callback=progress_callback,
    config={"strategy": "brkga", "generations": 500}
)
```

### Multi-Sheet Nesting

When parts don't fit on one sheet:

```python
parts = [
    {"id": f"part_{i}", "polygon": large_polygon, "quantity": 1}
    for i in range(50)
]

# Will automatically use multiple sheets
result = u_nesting.solve_2d(
    parts,
    {"width": 1000, "height": 500},
    config={"strategy": "brkga"}
)

print(f"Used {result['sheets_used']} sheets")
for p in result['placements']:
    print(f"  {p['id']} -> sheet {p['sheet_index']}")
```

## Tips and Best Practices

### 1. Simplify Polygons

Complex polygons with many vertices slow down computation. Simplify when possible:

```python
# Instead of 100+ vertices for a curve
# Use polygon approximation with fewer points
simplified = simplify_polygon(original, tolerance=1.0)
```

### 2. Use Appropriate Strategy

| Scenario | Recommended |
|----------|-------------|
| Quick preview | `blf` |
| Production quality | `brkga` |
| Very complex shapes | `nfp` + `brkga` |
| Large quantity | `alns` |

### 3. Set Time Limits

For interactive applications:

```python
config = {
    "strategy": "brkga",
    "time_limit_ms": 5000  # 5 seconds max
}
```

### 4. Handle Unplaced Items

```python
result = u_nesting.solve_2d(parts, sheet, config)

if result['unplaced']:
    print(f"Warning: {len(result['unplaced'])} parts didn't fit")
    for part_id in result['unplaced']:
        print(f"  - {part_id}")
```

### 5. Spacing for Tool Kerf

Account for cutting tool width:

```python
config = {
    "spacing": kerf_width + safety_margin  # e.g., 3.0 mm
}
```
