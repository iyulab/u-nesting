# 2D Nesting Guide

This guide covers the 2D nesting (polygon placement) functionality of U-Nesting.

## Overview

2D nesting optimizes the placement of irregular polygons on a sheet to minimize material waste. Common applications include:

- Sheet metal cutting
- Textile/leather cutting
- PCB panel arrangement
- Laser/waterjet cutting optimization

The request below is the JSON the C interface, the WebAssembly package and the
.NET package send; the Python package takes the same `geometries`, `boundary`
and `config` as dictionaries. Every key a request takes is listed here and in the
[JSON schemas](../json-schema/README.md). A key the engine does not read is
refused (`malformed_input`), never ignored.

```json
{
  "geometries": [
    {"id": "bracket", "polygon": [[0, 0], [60, 0], [60, 20], [20, 20], [20, 60], [0, 60]],
     "quantity": 8, "rotations": [0, 90, 180, 270]},
    {"id": "washer", "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]],
     "holes": [[[15, 15], [15, 35], [35, 35], [35, 15]]], "quantity": 4}
  ],
  "boundary": {"width": 400, "height": 300},
  "config": {"strategy": "nfp", "spacing": 3.0, "margin": 5.0, "time_limit_ms": 10000}
}
```

## Input Format

### Geometry Definition

Each geometry has:

<!-- fragment -->
```json
{
  "id": "unique_identifier",
  "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
  "quantity": 5,
  "rotations": [0, 90, 180, 270],
  "allow_flip": false,
  "holes": [[[40, 20], [40, 30], [60, 30], [60, 20]]]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier; two geometries with one id are refused (`duplicate_id`) |
| `polygon` | number[][] | Yes | Outer contour vertices (CCW) |
| `quantity` | integer | No | Number of copies, at least 1 (default: 1) |
| `rotations` | number[] | No | Allowed angles in degrees, at least one; omit it for a fixed orientation |
| `allow_flip` | boolean | No | Allow mirror reflection (default: false) |
| `holes` | number[][][] | No | Interior holes (CW orientation) |

### Wire Contract: Polylines and Units

- **Polylines only** — every contour is a flat array of vertices. Curves (arcs, béziers) are **not** accepted on the wire; flatten them to vertices with a tolerance on the consumer side before submitting (see [Simplify Polygons](#1-simplify-polygons)).
- **Units are consumer-defined** — all coordinates are unitless. Pick a unit (e.g. mm), and use the *same* unit across every field of a request (`polygon`, `holes`, `boundary`, `spacing`, `margin`). U-Nesting does no unit conversion.

### Polygon Orientation

- **Outer contour**: Counter-clockwise (CCW)
- **Holes**: Clockwise (CW)

```text
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

A rectangle, by `width` and `height`:

<!-- fragment -->
```json
{
  "width": 1000,
  "height": 500
}
```

or any polygon:

<!-- fragment -->
```json
{
  "polygon": [[0, 0], [1000, 0], [1000, 500], [400, 500], [0, 300]]
}
```

Give one shape: both, or half of a rectangle, is refused (`invalid_boundary`). A
side of 0 or below is refused (`parameter_out_of_range`, `boundary.width`).
There is one sheet; `config.multi_sheet` spreads what does not fit onto further
sheets of the same size (see [Multi-Sheet Nesting](#multi-sheet-nesting)).

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategy` | string | `"blf"` | `blf`, `nfp`, `ga`, `brkga`, `sa`, `gdrr` or `alns` (see the [Algorithm Reference](./algorithms.md)) |
| `spacing` | number | 0 | Minimum distance between parts, at least 0 |
| `margin` | number | 0 | Minimum distance from the sheet edges, at least 0 |
| `time_limit_ms` | integer | 30000 | Maximum solving time, at least 0 (0 = no limit) |
| `target_utilization` | number | none | `ga`, `brkga`, `sa`: stop once every part is placed at this utilization or more, in [0, 1] |
| `population_size` | integer | 30 | GA/BRKGA population size, at least 2; a value you set is run as given |
| `max_generations` | integer | 50 | GA/BRKGA generation budget, at least 0; a value you set is run as given |
| `crossover_rate` | number | 0.85 | GA crossover probability, in [0, 1] |
| `mutation_rate` | number | 0.05 | GA mutation probability, in [0, 1] |

A setting that only some strategies read (`target_utilization`, `population_size`, `max_generations`, `crossover_rate`, `mutation_rate`, `seed`) is refused as `invalid_option` when the request's strategy does not read it, with the strategies that do in the message -- it is not accepted and ignored.
| `seed` | integer | none | RNG seed of `ga`, `brkga`, `sa`, `gdrr`, `alns`: the same request and seed give the same layout when the run ends on its own budget, not on the clock (a search keeps a 10-15 s budget of its own when `time_limit_ms` is 0) |
| `multi_sheet` | boolean | false | Spill overflow onto additional sheets (sheet-local coords; `sheet_index` selects the sheet) |

The settings of one strategy are not read by another: `population_size` and the
rates mean nothing to `blf`. Nothing else is configurable through a request;
the [Performance Guide](./performance.md#algorithm-specific-tuning) says what
each search runs with.

## Output Format

The C interface and WebAssembly response:

<!-- fragment -->
```json
{
  "version": "0.18.0",
  "success": true,
  "error": null,
  "code": null,
  "details": null,
  "placements": [
    {
      "id": "bracket",
      "instance": 0,
      "x": 10.5,
      "y": 20.0,
      "rotation": 90,
      "sheet_index": 0,
      "flipped": false
    }
  ],
  "sheets_used": 1,
  "utilization": 0.82,
  "total_requested": 12,
  "unplaced": ["washer"],
  "unplaced_count": 1,
  "all_placed": false,
  "used_bounding_box": [380.0, 290.0],
  "used_utilization": 0.71,
  "stop_reason": "time_limit",
  "elapsed_ms": 1500
}
```

| Field | Description |
|-------|-------------|
| `success` | Whether the request was solved; `false` with `error`, `code` and `details` when it was refused |
| `placements` | List of part placements (instance-level) |
| `sheets_used` | Number of sheets used |
| `utilization` | Material utilization (0-1) |
| `total_requested` | Σ of every part's quantity (instance-level) |
| `unplaced` | **Deduplicated** IDs of parts that couldn't be placed (not per-instance) |
| `unplaced_count` | Instance-level count of parts that couldn't be placed |
| `all_placed` | Whether every requested instance was placed |
| `used_bounding_box`, `used_utilization` | The placed parts' footprint and their area over it |
| `stop_reason` | Why the run stopped: `completed`, `optimal`, `target_reached`, `iteration_limit`, `no_improvement`, `time_limit` (the best layout found when `time_limit_ms` ran out) or `cancelled` |
| `elapsed_ms` | Solving time in milliseconds |

The Python result has the same values under its own names: `geometry_id`,
`position` and `rotation` (radians) in each placement, `boundary_index` for the
sheet, `boundaries_used` and `computation_time_ms` (see the
[Python README](../../crates/python/README.md)).

### Placement Fields

| Field | Description |
|-------|-------------|
| `id` | Geometry ID |
| `instance` | Which copy of the geometry (0-based) |
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
        [[15, 15], [15, 35], [35, 35], [35, 15]]  # Inner hole (CW)
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
        [[10, 10], [10, 70], [90, 70], [90, 10]]
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

Progress is reported through the C interface
(`unesting_solve_2d_with_progress`, see the [FFI guide](./ffi-integration.md))
and the .NET package (`SolveWithProgress`, `SolveAsync` with `IProgress`, see
the [C# README](../../bindings/csharp/README.md)). `blf`, `nfp` and `ga` report
as they run; the other strategies run without reports. A progress
callback only observes the solve: the same request and seed give the same
layout with or without one.

### Multi-Sheet Nesting

When parts don't fit on one sheet, `multi_sheet` spills them onto further
sheets of the same size instead of leaving them unplaced:

```python
parts = [
    {"id": f"part_{i}", "polygon": [[0, 0], [300, 0], [300, 200], [0, 200]], "quantity": 1}
    for i in range(20)
]

result = u_nesting.solve_2d(
    parts,
    {"width": 1000, "height": 500},
    config={"strategy": "blf", "multi_sheet": True}
)

print(f"Used {result['boundaries_used']} sheets")
for p in result['placements']:
    print(f"  {p['geometry_id']} -> sheet {p['boundary_index']}")
```

Positions are sheet-local: group placements by sheet to draw one panel per
sheet.

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
| Irregular shapes | `nfp` |
| Production quality | `ga` or `brkga` |
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

if not result['success']:
    print(f"Refused ({result['code']}): {result['error']}")
elif not result['all_placed']:
    print(f"Warning: {result['unplaced_count']} parts didn't fit")
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
