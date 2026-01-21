# U-Nesting JSON Schema

This directory contains JSON Schema definitions for the U-Nesting FFI API.

## Schema Files

| File | Description |
|------|-------------|
| `request-2d.schema.json` | Request schema for 2D polygon nesting |
| `request-3d.schema.json` | Request schema for 3D bin packing |
| `response.schema.json` | Response schema for all operations |

## Usage

### Validation

You can validate your JSON requests against these schemas using any JSON Schema validator:

```bash
# Using ajv-cli (npm install -g ajv-cli)
ajv validate -s request-2d.schema.json -d my-request.json
```

### IDE Support

Most IDEs support JSON Schema for autocompletion and validation. Add a `$schema` property to your JSON files:

```json
{
  "$schema": "https://github.com/iyulab/U-Nesting/docs/json-schema/request-2d.schema.json",
  "geometries": [...],
  "boundary": {...}
}
```

## Quick Reference

### 2D Request Example

```json
{
  "mode": "2d",
  "geometries": [
    {
      "id": "part1",
      "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
      "quantity": 5,
      "rotations": [0, 90, 180, 270],
      "allow_flip": false
    }
  ],
  "boundary": {
    "width": 1000,
    "height": 500
  },
  "config": {
    "strategy": "nfp",
    "spacing": 2.0,
    "margin": 5.0,
    "time_limit_ms": 30000
  }
}
```

### 3D Request Example

```json
{
  "mode": "3d",
  "geometries": [
    {
      "id": "box1",
      "dimensions": [100, 50, 30],
      "quantity": 10,
      "mass": 2.5,
      "orientation": "upright"
    }
  ],
  "boundary": {
    "dimensions": [500, 400, 300],
    "max_mass": 100.0,
    "gravity": true,
    "stability": true
  },
  "config": {
    "strategy": "ep",
    "time_limit_ms": 30000
  }
}
```

### Response Example

```json
{
  "success": true,
  "placements": [
    {
      "geometry_id": "part1",
      "instance": 0,
      "position": [10.0, 20.0],
      "rotation": [1.5708],
      "boundary_index": 0
    }
  ],
  "boundaries_used": 1,
  "utilization": 0.85,
  "unplaced": [],
  "computation_time_ms": 1234
}
```

## Strategy Options

### 2D Strategies

| Strategy | Speed | Quality | Description |
|----------|-------|---------|-------------|
| `blf` | Fast | Basic | Bottom-Left Fill heuristic |
| `nfp` | Medium | Good | NFP-guided placement |
| `ga` | Slow | High | Genetic Algorithm |
| `brkga` | Medium | High | Biased Random-Key GA |
| `sa` | Medium | High | Simulated Annealing |

### 3D Strategies

| Strategy | Speed | Quality | Description |
|----------|-------|---------|-------------|
| `blf` | Fast | Basic | Bottom-Left Fill (layer-based) |
| `ep` | Fast | Good | Extreme Point heuristic |
| `ga` | Slow | High | Genetic Algorithm |
| `brkga` | Medium | High | Biased Random-Key GA |
| `sa` | Medium | High | Simulated Annealing |

## Orientation Constraints (3D)

| Value | Allowed Rotations | Description |
|-------|-------------------|-------------|
| `any` | 6 | Any axis-aligned orientation |
| `upright` | 2 | Original + 90° around Z axis |
| `fixed` | 1 | No rotation allowed |
