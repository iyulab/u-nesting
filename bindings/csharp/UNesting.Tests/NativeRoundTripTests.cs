using UNesting.Models;
using Xunit;

namespace UNesting.Tests;

/// <summary>
/// The solvers end to end against the native library, with reflection-based JSON disabled:
/// the request goes out and the result and progress reports come back through the client's
/// source-generated context.
/// </summary>
public class NativeRoundTripTests
{
    [Fact]
    public void Nesting_solves_and_reads_the_result()
    {
        using var nester = new Nester2D();
        var result = nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 40, 20, 2), Geometry2D.Rectangle("b", 30, 30)],
            Boundary = new Boundary2D { Width = 100, Height = 100 },
        });

        Assert.True(result.Success);
        Assert.Equal(3, result.TotalRequested);
        Assert.Equal(3, result.Placements.Count);
        Assert.True(result.AllPlaced);
        Assert.InRange(result.Utilization, 0.0, 1.0);
    }

    [Fact]
    public void Nesting_reports_progress()
    {
        using var nester = new Nester2D();
        var reports = 0;
        nester.ProgressChanged += (_, _) => reports++;
        var result = nester.SolveWithProgress(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 6)],
            Boundary = new Boundary2D { Width = 50, Height = 50 },
            Config = new Config2D { Strategy = "ga", PopulationSize = 10, MaxGenerations = 5, Seed = 1 },
        });

        Assert.True(result.Success);
        Assert.True(reports > 0, "no progress report was read");
    }

    [Fact]
    public void A_refusal_says_why_and_where()
    {
        using var nester = new Nester2D();

        var twice = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("plate", 10, 10), Geometry2D.Rectangle("rib", 5, 5),
                          Geometry2D.Rectangle("plate", 20, 20)],
            Boundary = new Boundary2D { Width = 100, Height = 100 },
        }));
        Assert.Equal("duplicate_id", twice.Reason);
        Assert.Equal("plate", twice.Details!.Value.GetProperty("id").GetString());
        Assert.Equal(0, twice.Details!.Value.GetProperty("first").GetInt32());
        Assert.Equal(2, twice.Details!.Value.GetProperty("index").GetInt32());
        Assert.Contains("given twice", twice.Message);

        var spacing = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = new Boundary2D { Width = 100, Height = 100 },
            Config = new Config2D { Spacing = -1 },
        }));
        Assert.Equal("parameter_out_of_range", spacing.Reason);
        Assert.Equal("spacing", spacing.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(-1.0, spacing.Details!.Value.GetProperty("got").GetDouble());

        var strategy = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = new Boundary2D { Width = 100, Height = 100 },
            Config = new Config2D { Strategy = "tabu" },
        }));
        Assert.Equal("unknown_option", strategy.Reason);
        Assert.Equal("tabu", strategy.Details!.Value.GetProperty("got").GetString());
    }

    [Fact]
    public void Packing_solves_and_reads_the_result()
    {
        using var packer = new Packer3D();
        var result = packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("box", 10, 10, 10, 4)],
            Boundary = new Boundary3D { Dimensions = [40, 40, 40] },
        });

        Assert.True(result.Success);
        Assert.Equal(4, result.Placements.Count);
    }
}
