# Getting Started

## Installation

### Python

```bash
pip install u-nesting
```

### C# / .NET

```bash
dotnet add package UNesting
```

### Rust (Native)

Add to `Cargo.toml`:

```toml
[dependencies]
u-nesting-d2 = "0.1"  # For 2D nesting
u-nesting-d3 = "0.1"  # For 3D packing
```

### C/C++ (FFI)

Build the FFI library:

```bash
cargo build -p u-nesting-ffi --release
```

Include the generated header:

```c
#include "u_nesting.h"
```

## First Example: 2D Nesting

### Python

```python
import u_nesting

# Define parts to nest
parts = [
    {
        "id": "rect1",
        "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
        "quantity": 5
    },
    {
        "id": "triangle",
        "polygon": [[0, 0], [80, 0], [40, 60]],
        "quantity": 3
    }
]

# Define sheet
sheet = {"width": 500, "height": 300}

# Solve
result = u_nesting.solve_2d(parts, sheet)

print(f"Utilization: {result['utilization']:.1%}")
for p in result['placements']:
    print(f"  {p['id']}: ({p['x']:.1f}, {p['y']:.1f}) @ {p['rotation']}°")
```

### C#

```csharp
using UNesting;
using UNesting.Models;

using var nester = new Nester2D();

var request = new NestingRequest
{
    Geometries = new List<Geometry2D>
    {
        Geometry2D.Rectangle("rect1", 100, 50, quantity: 5),
        new Geometry2D
        {
            Id = "triangle",
            Polygon = new[] {
                new[] { 0.0, 0.0 },
                new[] { 80.0, 0.0 },
                new[] { 40.0, 60.0 }
            },
            Quantity = 3
        }
    },
    Boundary = new Boundary2D { Width = 500, Height = 300 }
};

var result = nester.Solve(request);
Console.WriteLine($"Utilization: {result.Utilization:P1}");
```

## First Example: 3D Bin Packing

### Python

```python
import u_nesting

# Define boxes to pack
boxes = [
    {"id": "small", "dimensions": [20, 20, 20], "quantity": 10},
    {"id": "medium", "dimensions": [40, 30, 25], "quantity": 5},
    {"id": "large", "dimensions": [60, 40, 30], "quantity": 3}
]

# Define container
container = {
    "dimensions": [200, 150, 100],
    "gravity": True,
    "stability": True
}

# Solve
result = u_nesting.solve_3d(boxes, container)

print(f"Volume utilization: {result['utilization']:.1%}")
print(f"Bins used: {result['bins_used']}")
```

### C#

```csharp
using UNesting;
using UNesting.Models;

using var packer = new Packer3D();

var request = new PackingRequest
{
    Geometries = new List<Geometry3D>
    {
        Geometry3D.Box("small", 20, 20, 20, quantity: 10),
        Geometry3D.Box("medium", 40, 30, 25, quantity: 5),
        Geometry3D.Box("large", 60, 40, 30, quantity: 3)
    },
    Boundary = new Boundary3D
    {
        Dimensions = new[] { 200.0, 150.0, 100.0 },
        Gravity = true,
        Stability = true
    }
};

var result = packer.Solve(request);
Console.WriteLine($"Volume utilization: {result.Utilization:P1}");
```

## Configuration Options

Both 2D and 3D solvers accept a configuration object:

```python
config = {
    "strategy": "ga",        # Algorithm strategy
    "time_limit_ms": 30000,  # Max solving time
    "spacing": 2.0,          # Part spacing (2D only)
    "margin": 5.0            # Sheet margin (2D only)
}

result = u_nesting.solve_2d(parts, sheet, config=config)
```

See [2D Nesting Guide](./nesting-2d.md) and [3D Bin Packing Guide](./packing-3d.md) for detailed configuration options.
