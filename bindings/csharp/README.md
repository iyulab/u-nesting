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
    Geometries =
    [
        Geometry2D.Rectangle("part1", 100, 50, quantity: 5),
        new Geometry2D
        {
            Id = "triangle",
            Polygon = [[0.0, 0.0], [80.0, 0.0], [40.0, 60.0]],
            Quantity = 3,
            Rotations = [0, 90, 180, 270],
        },
    ],
    Boundary = Boundary2D.Rectangle(500, 300),
    Config = new Config2D
    {
        Strategy = Strategy2D.BottomLeftFill,
        Spacing = 2.0,
    },
};

// Solve
var result = nester.Solve(request);

Console.WriteLine($"Utilization: {result.Utilization:P1}");
Console.WriteLine($"Sheets used: {result.SheetsUsed}");
if (!result.AllPlaced)
    Console.WriteLine($"{result.UnplacedCount} copies did not fit: {string.Join(", ", result.Unplaced)}");

foreach (var placement in result.Placements)
{
    Console.WriteLine($"  {placement.Id}#{placement.Instance}: ({placement.X}, {placement.Y}) @ {placement.Rotation}°");
}
```

A sheet of any shape is a `Boundary2D { Polygon = [[x, y], ...] }` instead of a rectangle.

### 3D Bin Packing

```csharp
using UNesting;
using UNesting.Models;

using var packer = new Packer3D();

var request = new PackingRequest
{
    Geometries =
    [
        Geometry3D.Box("small", width: 20, depth: 20, height: 20, quantity: 10),
        Geometry3D.Box("medium", width: 40, depth: 30, height: 25, quantity: 5),
        new Geometry3D
        {
            Id = "fridge", Width = 60, Depth = 40, Height = 90, Quantity = 3,
            Mass = 45, Orientation = OrientationConstraint.Upright,
        },
    ],
    Boundary = new Boundary3D
    {
        Width = 200, Depth = 150, Height = 100,
        MaxMass = 500,
        Gravity = true,
        Stability = true,
    },
    Config = new Config3D { Strategy = Strategy3D.ExtremePoint },
};

var result = packer.Solve(request);

Console.WriteLine($"Utilization: {result.Utilization:P1}");
Console.WriteLine($"Bins used: {result.BinsUsed}");

foreach (var placement in result.Placements)
{
    Console.WriteLine($"  {placement.Id}#{placement.Instance}: ({placement.X}, {placement.Y}, {placement.Z}) turned {placement.Orientation}");
}
```

Sizes are width (x), depth (y) and height (z, the vertical) — the order the engine reads
`dimensions` in. `Orientation` limits how a box may be turned: `Any` (the default, six
ways), `Upright` (height stays vertical) or `Fixed`. A placement's `Orientation` says how
it was turned: which of the box's axes (`x` width, `y` depth, `z` height) lies along the
container's x, y and z — `Yxz` is a quarter turn about the vertical.

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

### 2D Nesting (`Strategy2D`)

| Strategy | Description |
|----------|-------------|
| `BottomLeftFill` | Bottom-left fill (fast, good for simple shapes) — the default |
| `NfpGuided` | No-fit-polygon guided placement (high quality) |
| `GeneticAlgorithm` | Genetic algorithm (optimization) |
| `Brkga` | Biased random-key GA (advanced optimization) |
| `SimulatedAnnealing` | Simulated annealing (global optimization) |
| `Gdrr` | Goal-driven ruin and recreate |
| `Alns` | Adaptive large neighbourhood search |

### 3D Bin Packing (`Strategy3D`)

| Strategy | Description |
|----------|-------------|
| `BottomLeftFill` | Layer-by-layer bottom-left fill (fast) — the default |
| `ExtremePoint` | Extreme-point heuristic (high quality) |
| `GeneticAlgorithm` | Genetic algorithm |
| `Brkga` | Biased random-key GA |
| `SimulatedAnnealing` | Simulated annealing |

Each strategy reads only some settings (`TargetUtilization` is read by the genetic,
BRKGA and annealing searches, `Seed` by the stochastic ones, ...); a setting the chosen
strategy does not read is refused (`invalid_option`), not ignored. Ask the library which
before building a form, instead of restating the table:

```csharp
var catalog = UNestingInfo.Strategies;               // StrategyCatalog
bool shown = catalog.Reads(Strategy2D.NfpGuided, NestingSetting.TargetUtilization); // false
foreach (var s in catalog.TwoD)                      // StrategyInfo2D(Name, Reads)
    Console.WriteLine($"{s.Name}: {string.Join(", ", s.Reads)}");
```

The catalog is the table the refusal itself is made from, so the two cannot disagree.

A setting left `null` in `Config2D` / `Config3D` is not sent, so the engine's default
applies; `0` is a value (`MaxGenerations = 0` evaluates the first population only,
`TimeLimitMs = 0` runs without a time limit).

## Refusals

A request the engine cannot honour raises `NestingException`. `Message` is readable
text; `ErrorCode` is a stable code to branch on (the error body's `code`), `Parameter`
and `Index` say which value it is about, `Details` holds the values behind it, and
`Status` is the native status:

```csharp
try
{
    nester.Solve(request);
}
catch (NestingException ex) when (ex.ErrorCode == "duplicate_id")
{
    var id = ex.Details!.Value.GetProperty("id").GetString();
    var again = ex.Details!.Value.GetProperty("index").GetInt32();
    Console.WriteLine($"geometry '{id}' is given again at position {again}");
}
```

The codes and their fields are listed in the npm package README's *Refusals* section;
the .NET package reports the same ones. `parameter` is always the path of the value in
the request -- `config.spacing`, `boundary.width`, `geometries[0].quantity`. A NaN or infinite number in a request cannot be
written as JSON, so the client refuses it before calling the engine, with the reason
`value_not_finite`: `Details` has `parameter`, the path to the value (or to the array
holding it, with `index` its position there), and `id` when it belongs to a geometry —
`{ "parameter": "geometries[1].polygon[2]", "index": 1, "id": "b" }` for a vertex's y.

A progress handler that throws stops the solve; `SolveWithProgress` rethrows its exception.

## Trimming and NativeAOT

The solvers serialize requests and read results and progress reports through
source-generated serialization, with no reflection, and the package is marked
`IsAotCompatible`. They run unchanged in trimmed and NativeAOT applications.

Results (`NestingResult`, `PackingResult`, their placements and `ProgressInfo`) are
records read strictly: a key the engine did not send fails the call instead of reading as
zero.

## Platforms

The package carries the native library for `win-x64`, `linux-x64` and `linux-arm64`
(glibc 2.39 or later), `osx-x64` and `osx-arm64`; no separate install is needed.

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
