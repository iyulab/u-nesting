# UNesting

.NET bindings for the U-Nesting spatial optimization engine.

## Features

- **2D Nesting**: Optimize polygon placement on sheets
- **3D Bin Packing**: Optimize box placement in containers
- **Progress Reporting**: Real-time progress callbacks with cancellation support
- **Async/Await**: Full async support with `IProgress<T>` and `CancellationToken`
- **Cross-Platform**: Windows, Linux, and macOS support

## Installation

```bash
dotnet add package UNesting
```

## Quick Start

### 2D Nesting

```csharp
using UNesting;
using UNesting.Models;

// Create a nester
using var nester = new Nester2D();

// Define geometries
var request = new NestingRequest
{
    Geometries = new List<Geometry2D>
    {
        Geometry2D.Rectangle("part1", 100, 50, quantity: 5),
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
    Boundary = new Boundary2D { Width = 500, Height = 300 },
    Config = new Config2D
    {
        Strategy = "blf",
        Spacing = 2.0
    }
};

// Solve
var result = nester.Solve(request);

Console.WriteLine($"Utilization: {result.Utilization:P1}");
Console.WriteLine($"Sheets used: {result.SheetsUsed}");

foreach (var placement in result.Placements)
{
    Console.WriteLine($"  {placement.Id}: ({placement.X}, {placement.Y}) @ {placement.Rotation}°");
}
```

### 3D Bin Packing

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
    },
    Config = new Config3D { Strategy = "ep" }
};

var result = packer.Solve(request);

Console.WriteLine($"Utilization: {result.Utilization:P1}");
Console.WriteLine($"Bins used: {result.BinsUsed}");
```

### Progress Reporting

```csharp
using var nester = new Nester2D();

// Using events
nester.ProgressChanged += (sender, e) =>
{
    Console.WriteLine($"Progress: {e.Progress.Iteration}/{e.Progress.TotalIterations}");

    // Cancel if needed
    if (e.Progress.Utilization > 0.95)
        e.Cancel = true;
};

var result = nester.SolveWithProgress(request);
```

### Async with IProgress

```csharp
var progress = new Progress<ProgressInfo>(p =>
{
    Console.WriteLine($"Iteration {p.Iteration}: {p.Utilization:P1}");
});

using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

try
{
    var result = await nester.SolveAsync(request, progress, cts.Token);
}
catch (OperationCanceledException)
{
    Console.WriteLine("Solving was cancelled");
}
```

## Strategies

### 2D Nesting

| Strategy | Description |
|----------|-------------|
| `blf` | Bottom-Left Fill (fast, good for simple shapes) |
| `nfp` | NFP-guided placement (high quality) |
| `ga` | Genetic Algorithm (optimization) |
| `brkga` | Biased Random-Key GA (advanced optimization) |
| `sa` | Simulated Annealing (global optimization) |

### 3D Bin Packing

| Strategy | Description |
|----------|-------------|
| `blf` | Bottom-Left Fill (fast) |
| `ep` | Extreme Point heuristic (high quality) |
| `ga` | Genetic Algorithm |
| `brkga` | Biased Random-Key GA |
| `sa` | Simulated Annealing |

## Native Library

The native library (`u_nesting_ffi.dll` / `libu_nesting_ffi.so` / `libu_nesting_ffi.dylib`) must be available in your application's runtime directory or system PATH.

### Building Native Library

```bash
cd <u-nesting-repo>
cargo build -p u-nesting-ffi --release
```

The built library will be in `target/release/`.

## API Reference

### Nester2D

```csharp
public class Nester2D : IDisposable
{
    // Synchronous solving
    NestingResult Solve(NestingRequest request);

    // Solving with progress callback
    NestingResult SolveWithProgress(NestingRequest request, CancellationToken ct = default);

    // Async solving
    Task<NestingResult> SolveAsync(NestingRequest request, IProgress<ProgressInfo>? progress = null, CancellationToken ct = default);

    // Progress event
    event EventHandler<ProgressEventArgs>? ProgressChanged;
}
```

### Packer3D

```csharp
public class Packer3D : IDisposable
{
    // Synchronous solving
    PackingResult Solve(PackingRequest request);

    // Solving with progress callback
    PackingResult SolveWithProgress(PackingRequest request, CancellationToken ct = default);

    // Async solving
    Task<PackingResult> SolveAsync(PackingRequest request, IProgress<ProgressInfo>? progress = null, CancellationToken ct = default);

    // Progress event
    event EventHandler<ProgressEventArgs>? ProgressChanged;
}
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.
