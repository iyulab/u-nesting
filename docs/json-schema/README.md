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

Most IDEs support JSON Schema for autocompletion and validation. Associate the
schema with your request files in the editor's settings -- in VS Code,
`json.schemas` in `settings.json`:

<!-- fragment -->
```json
{
  "json.schemas": [
    {"fileMatch": ["*.nesting.json"], "url": "./docs/json-schema/request-2d.schema.json"}
  ]
}
```

Do not add a `$schema` key to the request itself: a request takes the keys the
schema lists and no others, and the engine refuses one it does not read
(`malformed_input`).

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

<!-- fragment -->
```json
{
  "version": "0.18.0",
  "success": true,
  "placements": [
    {
      "id": "part1",
      "instance": 0,
      "x": 10.0,
      "y": 20.0,
      "rotation": 90.0,
      "sheet_index": 0,
      "flipped": false
    }
  ],
  "sheets_used": 1,
  "utilization": 0.85,
  "total_requested": 1,
  "unplaced": [],
  "unplaced_count": 0,
  "all_placed": true,
  "used_bounding_box": [100.0, 50.0],
  "used_utilization": 1.0,
  "elapsed_ms": 1234
}
```

A refused request has `success` `false`, readable `error` text, a stable `code`
and the values behind it in `details` (`parameter`, `id`, `index`, `min`, `max`,
`got`, ...); branch on `code`, not on the text.

`total_requested` is the Σ of every geometry's `quantity` (instance-level). Because
`unplaced` lists **deduplicated** geometry IDs, the number of unplaced instances is
`total_requested - placements.length`, not `unplaced.length`.

### Multi-Sheet Nesting (2D)

Set `config.multi_sheet: true` to distribute overflow across multiple sheets
instead of leaving it `unplaced`. `sheets_used` then reports the sheet count and
each placement's `sheet_index` selects its sheet. Placement `x`/`y` are
**sheet-local** (relative to each sheet's origin), so a placement on `sheet_index: 2`
is *not* offset by two sheet-widths — group placements by `sheet_index` to render
one panel per sheet. Default is `false` (single-sheet solve).

## Strategy Options

### 2D Strategies

| Strategy | Speed | Quality | Description |
|----------|-------|---------|-------------|
| `blf` | Fast | Basic | Bottom-Left Fill heuristic |
| `nfp` | Medium | Good | NFP-guided placement |
| `ga` | Slow | High | Genetic Algorithm |
| `brkga` | Medium | High | Biased Random-Key GA |
| `sa` | Medium | High | Simulated Annealing |
| `gdrr` | Medium | High | Goal-Driven Ruin and Recreate |
| `alns` | Medium | High | Adaptive Large Neighborhood Search |

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
