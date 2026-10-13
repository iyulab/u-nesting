// Run by scripts/nuget_package_smoke.sh against the freshly packed package:
// the native library loads, answers, and refuses with the documented surface.
using UNesting;
using UNesting.Models;

Console.WriteLine($"native version: {UNestingInfo.Version}");

var request = new NestingRequest
{
    Geometries = { Geometry2D.Rectangle("rect", 100, 50, quantity: 4) },
    Boundary = new Boundary2D { Width = 400, Height = 300 },
    Config = new Config2D { Strategy = Strategy2D.NfpGuided, TimeLimitMs = 5000 },
};

using var nester = new Nester2D();
var result = nester.Solve(request);

Console.WriteLine($"all_placed={result.AllPlaced} placed={result.Placements.Count} " +
                  $"sheets={result.SheetsUsed} utilization={result.Utilization:F4}");
if (!result.AllPlaced || result.Placements.Count != 4)
    throw new Exception($"expected 4 placements, got {result.Placements.Count}");

try
{
    nester.Solve(new NestingRequest
    {
        Geometries = { Geometry2D.Rectangle("rect", 100, 50) },
        Boundary = new Boundary2D { Width = 400, Height = 300 },
        Config = new Config2D { Spacing = -1 },
    });
    throw new Exception("a negative spacing was accepted");
}
catch (NestingException e) when (e.ErrorCode == "parameter_out_of_range" && e.Parameter == "config.spacing")
{
}

Console.WriteLine("nuget package smoke test passed");
