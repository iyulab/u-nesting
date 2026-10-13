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
            Boundary = Boundary2D.Rectangle(100, 100),
        });

        Assert.Equal(3, result.TotalRequested);
        Assert.Equal(3, result.Placements.Count);
        Assert.Equal([0, 1], result.Placements.Where(p => p.Id == "a").Select(p => p.Instance).Order());
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
            Boundary = Boundary2D.Rectangle(50, 50),
            Config = new Config2D { Strategy = Strategy2D.GeneticAlgorithm, PopulationSize = 10, MaxGenerations = 5, Seed = 1 },
        });

        Assert.True(result.AllPlaced);
        Assert.True(reports > 0, "no progress report was read");
    }

    [Fact]
    public void A_progress_handler_that_throws_stops_the_solve_with_its_exception()
    {
        using var packer = new Packer3D();
        packer.ProgressChanged += (_, _) => throw new InvalidOperationException("handler failed");
        var e = Assert.Throws<InvalidOperationException>(() => packer.SolveWithProgress(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, 10, 10, 4)],
            Boundary = Boundary3D.Box(30, 30, 30),
        }));
        Assert.Equal("handler failed", e.Message);
    }

    [Fact]
    public void A_progress_handler_can_cancel()
    {
        using var nester = new Nester2D();
        nester.ProgressChanged += (_, e) => e.Cancel = true;
        Assert.Throws<OperationCanceledException>(() => nester.SolveWithProgress(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 6)],
            Boundary = Boundary2D.Rectangle(50, 50),
            Config = new Config2D { Strategy = Strategy2D.GeneticAlgorithm, PopulationSize = 10, MaxGenerations = 50, Seed = 1 },
        }));
    }

    [Fact]
    public void A_refusal_says_why_and_where()
    {
        using var nester = new Nester2D();

        var twice = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("plate", 10, 10), Geometry2D.Rectangle("rib", 5, 5),
                          Geometry2D.Rectangle("plate", 20, 20)],
            Boundary = Boundary2D.Rectangle(100, 100),
        }));
        Assert.Equal("duplicate_id", twice.Reason);
        Assert.Equal("plate", twice.Details!.Value.GetProperty("id").GetString());
        Assert.Equal(0, twice.Details!.Value.GetProperty("first").GetInt32());
        Assert.Equal(2, twice.Details!.Value.GetProperty("index").GetInt32());
        Assert.Contains("given twice", twice.Message);

        var spacing = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = Boundary2D.Rectangle(100, 100),
            Config = new Config2D { Spacing = -1 },
        }));
        Assert.Equal("parameter_out_of_range", spacing.Reason);
        Assert.Equal("spacing", spacing.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(-1.0, spacing.Details!.Value.GetProperty("got").GetDouble());
        // 0 itself is accepted: a closed bound, so `min` and never `exclusive_min`.
        Assert.Equal(0.0, spacing.Details!.Value.GetProperty("min").GetDouble());
        Assert.False(spacing.Details!.Value.TryGetProperty("exclusive_min", out _));

        var shape = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = new Boundary2D { Width = 100, Height = 100, Polygon = [[0, 0], [100, 0], [0, 100]] },
        }));
        Assert.Equal("invalid_boundary", shape.Reason);
    }

    [Fact]
    public void A_negative_count_is_out_of_range()
    {
        using var nester = new Nester2D();

        var quantity = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("plate", 10, 10, -1)],
            Boundary = Boundary2D.Rectangle(100, 100),
        }));
        Assert.Equal("parameter_out_of_range", quantity.Reason);
        Assert.Equal("geometries[0].quantity", quantity.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal("plate", quantity.Details!.Value.GetProperty("id").GetString());
        Assert.Equal(1.0, quantity.Details!.Value.GetProperty("min").GetDouble());
        Assert.Equal(-1.0, quantity.Details!.Value.GetProperty("got").GetDouble());

        var population = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = Boundary2D.Rectangle(100, 100),
            Config = new Config2D { Strategy = Strategy2D.GeneticAlgorithm, PopulationSize = -1 },
        }));
        Assert.Equal("parameter_out_of_range", population.Reason);
        Assert.Equal("population_size", population.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(2.0, population.Details!.Value.GetProperty("min").GetDouble());
        Assert.Equal(-1.0, population.Details!.Value.GetProperty("got").GetDouble());
    }

    [Fact]
    public void Nothing_to_place_and_a_flat_boundary_are_refused()
    {
        using var nester = new Nester2D();
        var empty = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [],
            Boundary = Boundary2D.Rectangle(100, 100),
        }));
        Assert.Equal("empty_input", empty.Reason);
        Assert.Equal("geometries", empty.Details!.Value.GetProperty("parameter").GetString());

        var flat = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = Boundary2D.Rectangle(0, 100),
        }));
        Assert.Equal("parameter_out_of_range", flat.Reason);
        Assert.Equal("boundary.width", flat.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(0.0, flat.Details!.Value.GetProperty("exclusive_min").GetDouble());
        Assert.Equal(0.0, flat.Details!.Value.GetProperty("got").GetDouble());

        using var packer = new Packer3D();
        var none = Assert.Throws<NestingException>(() => packer.Solve(new PackingRequest
        {
            Geometries = [],
            Boundary = Boundary3D.Box(40, 40, 40),
        }));
        Assert.Equal("empty_input", none.Reason);
        Assert.Equal("geometries", none.Details!.Value.GetProperty("parameter").GetString());
    }

    [Fact]
    public void A_3D_measure_out_of_range_is_named_by_its_place()
    {
        using var packer = new Packer3D();

        var depth = Assert.Throws<NestingException>(() => packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, 10, 10)],
            Boundary = Boundary3D.Box(40, 0, 40),
        }));
        Assert.Equal("parameter_out_of_range", depth.Reason);
        Assert.Equal("boundary.dimensions[1]", depth.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(0.0, depth.Details!.Value.GetProperty("exclusive_min").GetDouble());

        var mass = Assert.Throws<NestingException>(() => packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, 10, 10)],
            Boundary = new Boundary3D { Width = 40, Depth = 40, Height = 40, MaxMass = 0 },
        }));
        Assert.Equal("boundary.max_mass", mass.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(0.0, mass.Details!.Value.GetProperty("exclusive_min").GetDouble());

        var height = Assert.Throws<NestingException>(() => packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("flat", 10, 10, -1)],
            Boundary = Boundary3D.Box(40, 40, 40),
        }));
        Assert.Equal("geometries[0].dimensions[2]", height.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal("flat", height.Details!.Value.GetProperty("id").GetString());
    }

    [Fact]
    public void A_non_finite_number_is_refused_where_it_sits()
    {
        using var nester = new Nester2D();
        var vertex = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10), Geometry2D.Rectangle("b", 10, double.NaN)],
            Boundary = Boundary2D.Rectangle(100, 100),
        }));
        Assert.Equal("value_not_finite", vertex.Reason);
        Assert.Equal(NativeLibrary.UNESTING_ERR_SOLVE_FAILED, vertex.ErrorCode);
        Assert.Equal("geometries[1].polygon[2]", vertex.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(1, vertex.Details!.Value.GetProperty("index").GetInt32());
        Assert.Equal("b", vertex.Details!.Value.GetProperty("id").GetString());

        var margin = Assert.Throws<NestingException>(() => nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10)],
            Boundary = Boundary2D.Rectangle(100, 100),
            Config = new Config2D { Margin = double.PositiveInfinity },
        }));
        Assert.Equal("value_not_finite", margin.Reason);
        Assert.Equal("config.margin", margin.Details!.Value.GetProperty("parameter").GetString());
        Assert.False(margin.Details!.Value.TryGetProperty("index", out _));

        using var packer = new Packer3D();
        var box = Assert.Throws<NestingException>(() => packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, double.NegativeInfinity, 10)],
            Boundary = Boundary3D.Box(40, 40, 40),
        }));
        Assert.Equal("geometries[0].dimensions", box.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(1, box.Details!.Value.GetProperty("index").GetInt32());

        var container = Assert.Throws<NestingException>(() => packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, 10, 10)],
            Boundary = Boundary3D.Box(40, 40, double.NaN),
        }));
        Assert.Equal("boundary.dimensions", container.Details!.Value.GetProperty("parameter").GetString());
        Assert.Equal(2, container.Details!.Value.GetProperty("index").GetInt32());
    }

    [Fact]
    public void A_budget_that_is_set_is_run_and_zero_generations_is_a_budget()
    {
        using var nester = new Nester2D();
        var result = nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 3)],
            Boundary = Boundary2D.Rectangle(50, 50),
            Config = new Config2D { Strategy = Strategy2D.Brkga, PopulationSize = 3, MaxGenerations = 0, Seed = 1 },
        });
        Assert.Equal(3, result.Placements.Count);
    }

    [Fact]
    public void Packing_solves_and_reads_the_result()
    {
        using var packer = new Packer3D();
        var result = packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("box", 10, 10, 10, 4)],
            Boundary = Boundary3D.Box(40, 40, 40),
        });

        Assert.True(result.AllPlaced);
        Assert.Equal(4, result.Placements.Count);
    }

    [Fact]
    public void A_box_is_read_width_depth_height()
    {
        using var packer = new Packer3D();
        // Held as given, a 30 x 10 x 5 box fits a 32 x 12 x 6 container exactly one way.
        var fits = packer.Solve(new PackingRequest
        {
            Geometries = [new Geometry3D { Id = "b", Width = 30, Depth = 10, Height = 5, Orientation = OrientationConstraint.Fixed }],
            Boundary = Boundary3D.Box(width: 32, depth: 12, height: 6),
        });
        Assert.True(fits.AllPlaced);
        Assert.Equal(AxisPermutation.Xyz, fits.Placements[0].Orientation);

        // The same container with depth and height swapped does not take it.
        var swapped = packer.Solve(new PackingRequest
        {
            Geometries = [new Geometry3D { Id = "b", Width = 30, Depth = 10, Height = 5, Orientation = OrientationConstraint.Fixed }],
            Boundary = Boundary3D.Box(width: 32, depth: 6, height: 12),
        });
        Assert.False(swapped.AllPlaced);

        // Upright keeps the height vertical: a quarter turn about it is all it may do.
        var upright = packer.Solve(new PackingRequest
        {
            Geometries = [new Geometry3D { Id = "b", Width = 30, Depth = 10, Height = 5, Orientation = OrientationConstraint.Upright }],
            Boundary = Boundary3D.Box(width: 12, depth: 32, height: 6),
        });
        Assert.True(upright.AllPlaced);
        Assert.Equal(AxisPermutation.Yxz, upright.Placements[0].Orientation);
    }
}
