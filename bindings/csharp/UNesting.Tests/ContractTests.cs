using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using UNesting.Models;
using Xunit;

namespace UNesting.Tests;

/// <summary>
/// The client and the engine say the same thing, both ways, against the real native library
/// with reflection-based serialization disabled.
/// <list type="bullet">
/// <item>Results: the raw response and the record written back out are the same JSON — no
/// key dropped, renamed or added, no value changed.</item>
/// <item>Requests: every key the client can write is one the engine's request schema has,
/// every key the schema has can be written (but for the ones listed as not needed), and a
/// request with every setting given is accepted and solved.</item>
/// </list>
/// </summary>
public class ContractTests
{
    // ── Results ──

    /// <summary>The envelope a returned result is read from; a refusal raises instead.</summary>
    private static readonly string[] Envelope = ["success", "error", "code", "details"];

    public static TheoryData<string> Solves() => [.. Cases.Keys];

    private static readonly Dictionary<string, Func<(string Raw, JsonNode? Written)>> Cases = new()
    {
        ["2D"] = () => Solve2D(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 40, 20, 2), Geometry2D.Rectangle("b", 30, 30)],
            Boundary = Boundary2D.Rectangle(100, 100),
        }),
        ["2D(unplaced)"] = () => Solve2D(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 40, 40, 3), Geometry2D.Rectangle("huge", 500, 10)],
            Boundary = Boundary2D.Rectangle(100, 50),
        }),
        ["2D(multi-sheet)"] = () => Solve2D(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 60, 60, 3)],
            Boundary = Boundary2D.Rectangle(100, 100),
            Config = new Config2D { MultiSheet = true },
        }),
        ["3D"] = () => Solve3D(new PackingRequest
        {
            Geometries = [Geometry3D.Box("box", 10, 20, 30, 4)],
            Boundary = Boundary3D.Box(40, 40, 40),
        }),
        ["3D(turned, unplaced)"] = () => Solve3D(new PackingRequest
        {
            Geometries = [Geometry3D.Box("long", 30, 10, 5, 2), Geometry3D.Box("huge", 100, 100, 100)],
            Boundary = Boundary3D.Box(12, 32, 40),
            Config = new Config3D { Strategy = Strategy3D.ExtremePoint },
        }),
    };

    private static NestingRequest LongSearch() => new()
    {
        Geometries = [Geometry2D.Rectangle("a", 7, 3, 150)],
        Boundary = Boundary2D.Rectangle(500, 5000),
        Config = new Config2D { Strategy = Strategy2D.Gdrr, TimeLimitMs = 120_000, Seed = 1 },
    };

    /// A cancel reaches a running search: the engine hears it through its
    /// progress callback, called as the search runs. It used to be called only
    /// before and after the solve, so a cancel waited out the search.
    [Fact]
    public void A_cancel_reaches_a_running_search()
    {
        using var nester = new Nester2D();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));
        var watch = System.Diagnostics.Stopwatch.StartNew();
        Assert.Throws<OperationCanceledException>(() => nester.SolveWithProgress(LongSearch(), cts.Token));
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5), $"{watch.Elapsed} after a cancel at 0.5 s");
    }

    /// The same through <see cref="Nester2D.SolveAsync"/> without a progress
    /// reporter, which used to run the solve without a callback at all.
    [Fact]
    public async Task A_cancel_reaches_an_async_solve_without_a_reporter()
    {
        using var nester = new Nester2D();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));
        var watch = System.Diagnostics.Stopwatch.StartNew();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => nester.SolveAsync(LongSearch(), cancellationToken: cts.Token));
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5), $"{watch.Elapsed} after a cancel at 0.5 s");
    }

    [Fact]
    public void A_solve_says_why_it_stopped()
    {
        using var nester = new Nester2D();
        var fill = nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 2)],
            Boundary = Boundary2D.Rectangle(50, 50),
            Config = new Config2D { Strategy = Strategy2D.BottomLeftFill },
        });
        Assert.Equal(StopReason.Completed, fill.StopReason);

        var search = nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 2)],
            Boundary = Boundary2D.Rectangle(50, 50),
            Config = new Config2D { Strategy = Strategy2D.GeneticAlgorithm, TargetUtilization = 0, Seed = 1 },
        });
        Assert.Equal(StopReason.TargetReached, search.StopReason);

        using var packer = new Packer3D();
        var packed = packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, 10, 10, 2)],
            Boundary = Boundary3D.Box(30, 30, 30),
            Config = new Config3D { Strategy = Strategy3D.ExtremePoint },
        });
        Assert.Equal(StopReason.Completed, packed.StopReason);
    }

    private static (string, JsonNode?) Solve2D(NestingRequest request)
    {
        using var nester = new Nester2D();
        string? raw = null;
        nester.Observers.Response = body => raw = body;
        var result = nester.Solve(request);
        return (raw!, JsonSerializer.SerializeToNode(result, NestingJson.Default.NestingResult));
    }

    private static (string, JsonNode?) Solve3D(PackingRequest request)
    {
        using var packer = new Packer3D();
        string? raw = null;
        packer.Observers.Response = body => raw = body;
        var result = packer.Solve(request);
        return (raw!, JsonSerializer.SerializeToNode(result, NestingJson.Default.PackingResult));
    }

    [Theory]
    [MemberData(nameof(Solves))]
    public void The_result_carries_exactly_what_the_engine_sent(string solve)
    {
        var (raw, written) = Cases[solve]();

        var engine = JsonNode.Parse(raw)!.AsObject();
        // The envelope of a returned result says success and nothing else.
        Assert.True(engine["success"]!.GetValue<bool>());
        foreach (var key in Envelope[1..])
            Assert.True(engine.ContainsKey(key) && engine[key] is null, $"{key}: {engine[key]?.ToJsonString()}");
        foreach (var key in Envelope)
            engine.Remove(key);

        var difference = Difference(engine, written, "$");
        Assert.True(difference is null, $"{solve}: {difference}\nengine: {raw}\nrecord: {written?.ToJsonString()}");
    }

    [Fact]
    public void A_layout_is_checked_and_the_check_carries_exactly_what_the_engine_sent()
    {
        var request = new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 2)],
            Boundary = Boundary2D.Rectangle(40, 20),
            Config = new Config2D { Spacing = 1 },
        };
        using var nester = new Nester2D();
        var solved = nester.Solve(request);
        var clean = nester.Check(request, solved);
        Assert.True(clean.Feasible);
        Assert.Empty(clean.Violations);

        string? raw = null;
        nester.Observers.Response = body => raw = body;
        var stacked = solved with
        {
            Placements = [solved.Placements[0], solved.Placements[1] with { X = solved.Placements[0].X, Y = solved.Placements[0].Y }],
        };
        var check = nester.Check(request, stacked);
        var engine = JsonNode.Parse(raw!)!.AsObject();
        Assert.True(engine["success"]!.GetValue<bool>());
        foreach (var key in Envelope)
            engine.Remove(key);
        var written = JsonSerializer.SerializeToNode(check, NestingJson.Default.LayoutCheck);
        var difference = Difference(engine, written, "$");
        Assert.True(difference is null, $"{difference}\nengine: {raw}\nrecord: {written?.ToJsonString()}");
        Assert.False(check.Feasible);
        Assert.Equal([new LayoutViolation.Overlap(0, 1)], check.Violations);
    }

    [Fact]
    public void A_packing_is_checked_and_a_floating_box_is_unsupported()
    {
        var request = new PackingRequest
        {
            Geometries = [Geometry3D.Box("box", 10, 10, 10, 2)],
            Boundary = new Boundary3D { Width = 30, Depth = 20, Height = 20, Gravity = true },
        };
        using var packer = new Packer3D();
        var packed = packer.Solve(request);
        Assert.True(packer.Check(request, packed).Feasible);

        string? raw = null;
        packer.Observers.Response = body => raw = body;
        var lifted = packed with { Placements = [packed.Placements[0] with { Z = 5 }, .. packed.Placements.Skip(1)] };
        var check = packer.Check(request, lifted);
        var engine = JsonNode.Parse(raw!)!.AsObject();
        foreach (var key in Envelope)
            engine.Remove(key);
        var written = JsonSerializer.SerializeToNode(check, NestingJson.Default.LayoutCheck);
        var difference = Difference(engine, written, "$");
        Assert.True(difference is null, $"{difference}\nengine: {raw}\nrecord: {written?.ToJsonString()}");
        Assert.Contains(new LayoutViolation.Unsupported(0, "box"), check.Violations);
    }

    [Fact]
    public void The_strategy_catalog_carries_exactly_what_the_engine_sent()
    {
        var raw = UNestingInfo.StrategiesJson;
        var written = JsonSerializer.SerializeToNode(UNestingInfo.Strategies, NestingJson.Default.StrategyCatalog);
        var difference = Difference(JsonNode.Parse(raw), written, "$");
        Assert.True(difference is null, $"{difference}\nengine: {raw}\nrecord: {written?.ToJsonString()}");

        // The enums are the strategies the engine offers, each dimension.
        var catalog = UNestingInfo.Strategies;
        Assert.Equal(Enum.GetValues<Strategy2D>().Order(), catalog.TwoD.Select(s => s.Name).Order());
        Assert.Equal(Enum.GetValues<Strategy3D>().Order(), catalog.ThreeD.Select(s => s.Name).Order());
        Assert.True(catalog.Reads(Strategy2D.GeneticAlgorithm, NestingSetting.TargetUtilization));
        Assert.False(catalog.Reads(Strategy2D.NfpGuided, NestingSetting.TargetUtilization));
        Assert.False(catalog.Reads(Strategy3D.BottomLeftFill, NestingSetting.MultiSheet));
    }

    [Fact]
    public void A_turned_box_reads_its_orientation()
    {
        // 30 wide does not fit 12 wide: the box can only go in turned.
        var (_, written) = Cases["3D(turned, unplaced)"]();
        var orientations = written!["placements"]!.AsArray().Select(p => p!["orientation"]!.GetValue<string>());
        Assert.Contains(orientations, o => o != "xyz");
    }

    public static TheoryData<string> Progresses() => ["2D", "3D"];

    [Theory]
    [MemberData(nameof(Progresses))]
    public void Every_progress_report_carries_exactly_what_the_engine_sent(string dimension)
    {
        var raws = new List<string>();
        var reports = new List<ProgressInfo>();
        if (dimension == "2D")
        {
            using var nester = new Nester2D();
            nester.Observers.Progress = raws.Add;
            nester.ProgressChanged += (_, e) => reports.Add(e.Progress);
            nester.SolveWithProgress(new NestingRequest
            {
                Geometries = [Geometry2D.Rectangle("a", 10, 10, 6)],
                Boundary = Boundary2D.Rectangle(50, 50),
                Config = new Config2D { Strategy = Strategy2D.GeneticAlgorithm, PopulationSize = 6, MaxGenerations = 3, Seed = 1 },
            });
        }
        else
        {
            using var packer = new Packer3D();
            packer.Observers.Progress = raws.Add;
            packer.ProgressChanged += (_, e) => reports.Add(e.Progress);
            packer.SolveWithProgress(new PackingRequest
            {
                Geometries = [Geometry3D.Box("b", 10, 10, 10, 4)],
                Boundary = Boundary3D.Box(30, 30, 30),
                Config = new Config3D { Strategy = Strategy3D.GeneticAlgorithm, PopulationSize = 6, MaxGenerations = 3, Seed = 1 },
            });
        }

        Assert.NotEmpty(raws);
        Assert.Equal(raws.Count, reports.Count);
        for (var i = 0; i < raws.Count; i++)
        {
            var written = JsonSerializer.SerializeToNode(reports[i], NestingJson.Default.ProgressInfo);
            var difference = Difference(JsonNode.Parse(raws[i]), written, "$");
            Assert.True(difference is null, $"report {i}: {difference}\nengine: {raws[i]}\nrecord: {written?.ToJsonString()}");
        }
    }

    [Fact]
    public void A_missing_key_fails_rather_than_reading_as_zero()
    {
        Assert.ThrowsAny<JsonException>(() => JsonSerializer.Deserialize(
            """{"version": "0", "placements": [], "sheets_used": 1, "utilization": 0, "unplaced": [], "unplaced_count": 0, "all_placed": true, "used_bounding_box": [0, 0], "used_utilization": 0, "elapsed_ms": 0}""",
            NestingJson.Default.NestingResult));
        Assert.ThrowsAny<JsonException>(() => JsonSerializer.Deserialize(
            """{"id": "b", "instance": 0, "bin_index": 0, "x": 0, "y": 0, "z": 0, "orientation": "sideways"}""",
            NestingJson.Default.Placement3D));
        Assert.ThrowsAny<JsonException>(() => JsonSerializer.Deserialize(
            """{"iteration": 0, "total_iterations": 0, "utilization": 0, "best_fitness": 0, "items_placed": 0, "total_items": 1, "elapsed_ms": 0, "phase": null, "running": true}""",
            NestingJson.Default.ProgressInfo));
    }

    // ── Requests ──

    /// <summary>Request keys the client does not write: it calls the per-dimension entry points, so neither the auto-detect discriminator nor the version tag (which the engine does not read) has a use.</summary>
    private static readonly HashSet<string> NotNeeded = ["$.mode", "$.version"];

    private static NestingRequest Full2D(Boundary2D boundary) => new()
    {
        Geometries =
        [
            new Geometry2D
            {
                Id = "frame",
                Polygon = [[0, 0], [30, 0], [30, 30], [0, 30]],
                Holes = [[[10, 10], [20, 10], [20, 20], [10, 20]]],
                Quantity = 2,
                Rotations = [0, 90],
                AllowFlip = true,
            },
        ],
        Boundary = boundary,
        Config = new Config2D
        {
            Strategy = Strategy2D.GeneticAlgorithm,
            Spacing = 1,
            Margin = 1,
            TimeLimitMs = 5000,
            TargetUtilization = 0.99,
            PopulationSize = 4,
            MaxGenerations = 2,
            CrossoverRate = 0.8,
            MutationRate = 0.1,
            Seed = 7,
            MultiSheet = true,
        },
    };

    private static readonly NestingRequest Full2DRectangle = Full2D(Boundary2D.Rectangle(100, 100));

    /// The iteration budget, which the population searches do not read: beside SA.
    private static readonly NestingRequest IterationBudget2D = new()
    {
        Geometries = [Geometry2D.Rectangle("a", 10, 10, 2)],
        Boundary = Boundary2D.Rectangle(100, 100),
        Config = new Config2D { Strategy = Strategy2D.SimulatedAnnealing, MaxIterations = 50, Seed = 7 },
    };

    private static readonly PackingRequest IterationBudget3D = new()
    {
        Geometries = [Geometry3D.Box("b", 10, 10, 10, 2)],
        Boundary = Boundary3D.Box(30, 30, 30),
        Config = new Config3D { Strategy = Strategy3D.SimulatedAnnealing, MaxIterations = 50, Seed = 7 },
    };
    private static readonly NestingRequest Full2DPolygon = Full2D(new Boundary2D { Polygon = [[0, 0], [100, 0], [100, 100], [0, 100]] });

    private static readonly PackingRequest Full3D = new()
    {
        Geometries =
        [
            new Geometry3D { Id = "crate", Width = 10, Depth = 20, Height = 5, Quantity = 3, Mass = 2, Orientation = OrientationConstraint.Upright },
        ],
        Boundary = new Boundary3D { Width = 40, Depth = 40, Height = 40, MaxMass = 100, Gravity = true, Stability = true },
        Config = new Config3D
        {
            Strategy = Strategy3D.GeneticAlgorithm,
            Spacing = 1,
            Margin = 1,
            TimeLimitMs = 5000,
            TargetUtilization = 0.99,
            PopulationSize = 4,
            MaxGenerations = 2,
            CrossoverRate = 0.8,
            MutationRate = 0.1,
            Seed = 7,
        },
    };

    [Fact]
    public void A_request_with_every_setting_is_accepted_and_solved()
    {
        using var nester = new Nester2D();
        foreach (var request in new[] { Full2DRectangle, Full2DPolygon, IterationBudget2D })
        {
            var result = nester.Solve(request);
            Assert.Equal(2, result.Placements.Count);
        }

        using var packer = new Packer3D();
        var packed = packer.Solve(Full3D);
        Assert.Equal(3, packed.TotalRequested);
        Assert.NotEmpty(packed.Placements);
        Assert.Equal(2, packer.Solve(IterationBudget3D).Placements.Count);
    }

    [Fact]
    public void The_2D_request_keys_are_the_engines()
    {
        var schema = Schema("request-2d");
        var written = Keys(Write(Full2DRectangle, NestingJson.Default.NestingRequest))
            .Union(Keys(Write(Full2DPolygon, NestingJson.Default.NestingRequest)))
            .Union(Keys(Write(IterationBudget2D, NestingJson.Default.NestingRequest)))
            .ToHashSet();
        Compare(schema, written);
    }

    [Fact]
    public void The_3D_request_keys_are_the_engines()
        => Compare(
            Schema("request-3d"),
            Keys(Write(Full3D, NestingJson.Default.PackingRequest))
                .Union(Keys(Write(IterationBudget3D, NestingJson.Default.PackingRequest)))
                .ToHashSet());

    [Fact]
    public void Each_option_is_a_name_the_engine_reads()
    {
        var d2 = SchemaRoot("request-2d");
        var d3 = SchemaRoot("request-3d");
        Assert.Subset(Literals(d2, "Config", "strategy").ToHashSet(), Names(Enum.GetValues<Strategy2D>(), NestingJson.Default.Strategy2D).ToHashSet());
        Assert.Subset(Literals(d3, "Config", "strategy").ToHashSet(), Names(Enum.GetValues<Strategy3D>(), NestingJson.Default.Strategy3D).ToHashSet());
        Assert.Equal(Literals(d3, "Geometry3D", "orientation"), Names(Enum.GetValues<OrientationConstraint>(), NestingJson.Default.OrientationConstraint));
    }

    // What each strategy reads (the engine's Strategy::reads_setting): a setting
    // a strategy does not read is refused, so each run carries only its own.
    private static bool Population(Strategy2D s) => s is Strategy2D.GeneticAlgorithm or Strategy2D.Brkga;
    private static bool Population(Strategy3D s) => s is Strategy3D.GeneticAlgorithm or Strategy3D.Brkga;
    private static bool Seeded(Strategy2D s) =>
        s is Strategy2D.GeneticAlgorithm or Strategy2D.Brkga or Strategy2D.SimulatedAnnealing or Strategy2D.Gdrr or Strategy2D.Alns;
    private static bool Seeded(Strategy3D s) =>
        s is Strategy3D.GeneticAlgorithm or Strategy3D.Brkga or Strategy3D.SimulatedAnnealing;

    public static TheoryData<Strategy2D> Strategies2D() => [.. Enum.GetValues<Strategy2D>()];
    public static TheoryData<Strategy3D> Strategies3D() => [.. Enum.GetValues<Strategy3D>()];

    [Theory]
    [MemberData(nameof(Strategies2D))]
    public void Every_2D_strategy_runs(Strategy2D strategy)
    {
        using var nester = new Nester2D();
        var result = nester.Solve(new NestingRequest
        {
            Geometries = [Geometry2D.Rectangle("a", 10, 10, 2)],
            Boundary = Boundary2D.Rectangle(50, 50),
            // The settings the strategy reads: one it does not read is refused.
            Config = new Config2D
            {
                Strategy = strategy,
                PopulationSize = Population(strategy) ? 10 : null,
                MaxGenerations = Population(strategy) ? 5 : null,
                Seed = Seeded(strategy) ? 1 : null,
                TimeLimitMs = 1000,
            },
        });
        Assert.True(result.AllPlaced);
    }

    [Theory]
    [MemberData(nameof(Strategies3D))]
    public void Every_3D_strategy_runs(Strategy3D strategy)
    {
        using var packer = new Packer3D();
        var result = packer.Solve(new PackingRequest
        {
            Geometries = [Geometry3D.Box("b", 10, 10, 10, 2)],
            Boundary = Boundary3D.Box(30, 30, 30),
            Config = new Config3D
            {
                Strategy = strategy,
                PopulationSize = Population(strategy) ? 10 : null,
                MaxGenerations = Population(strategy) ? 5 : null,
                Seed = Seeded(strategy) ? 1 : null,
                TimeLimitMs = 1000,
            },
        });
        Assert.True(result.AllPlaced);
    }

    [Fact]
    public void A_setting_left_null_is_left_out_and_zero_is_a_value()
    {
        Assert.Equal(["geometries", "boundary"], Write(new NestingRequest(), NestingJson.Default.NestingRequest).AsObject().Select(p => p.Key));
        Assert.Empty(Write(new Config2D(), NestingJson.Default.Config2D).AsObject());
        Assert.Empty(Write(new Config3D(), NestingJson.Default.Config3D).AsObject());
        Assert.Equal(["dimensions"], Write(new Boundary3D(), NestingJson.Default.Boundary3D).AsObject().Select(p => p.Key));

        // 0 generations and 0 ms (no limit) are values, not "not set".
        var set = Write(new Config2D { MaxGenerations = 0, TimeLimitMs = 0, Spacing = 0 }, NestingJson.Default.Config2D);
        Assert.Equal(0, set["max_generations"]!.GetValue<int>());
        Assert.Equal(0, set["time_limit_ms"]!.GetValue<int>());
        Assert.Equal(0.0, set["spacing"]!.GetValue<double>());
    }

    [Fact]
    public void A_box_is_written_width_depth_height()
    {
        var box = Write(Geometry3D.Box("b", width: 1, depth: 2, height: 3), NestingJson.Default.Geometry3D);
        Assert.Equal([1.0, 2.0, 3.0], box["dimensions"]!.AsArray().Select(v => v!.GetValue<double>()));
        var container = Write(Boundary3D.Box(width: 4, depth: 5, height: 6), NestingJson.Default.Boundary3D);
        Assert.Equal([4.0, 5.0, 6.0], container["dimensions"]!.AsArray().Select(v => v!.GetValue<double>()));
    }

    [Fact]
    public void The_suite_runs_as_a_trimmed_host_does()
    {
        Assert.False(JsonSerializer.IsReflectionEnabledByDefault);
        Assert.Throws<InvalidOperationException>(() => JsonSerializer.Serialize(new Config2D(), new JsonSerializerOptions()));
    }

    // ── Helpers ──

    private static JsonNode Write<T>(T value, JsonTypeInfo<T> info) => JsonSerializer.SerializeToNode(value, info)!;

    private static string[] Names<T>(T[] values, JsonTypeInfo<T> info)
        => [.. values.Select(v => JsonSerializer.SerializeToNode(v, info)!.GetValue<string>())];

    private static JsonObject SchemaRoot(string name)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "schema", $"{name}.schema.json");
        Assert.True(File.Exists(path), $"{path} is not copied next to the tests");
        return JsonNode.Parse(File.ReadAllText(path))!.AsObject();
    }

    /// <summary>Every key path the schema defines: <c>$.config.seed</c>, <c>$.geometries[].id</c>, ...</summary>
    private static HashSet<string> Schema(string name)
    {
        var root = SchemaRoot(name);
        var keys = new HashSet<string>();
        Walk(root, root, "$", keys);
        return keys;
    }

    private static void Walk(JsonObject root, JsonNode node, string path, HashSet<string> keys)
    {
        if (node["$ref"] is { } reference)
        {
            var target = root;
            foreach (var part in reference.GetValue<string>().TrimStart('#', '/').Split('/'))
                target = target[part]!.AsObject();
            Walk(root, target, path, keys);
            return;
        }
        if (node["items"] is { } items && node["type"]?.GetValue<string>() == "array")
            Walk(root, items, $"{path}[]", keys);
        if (node["oneOf"] is JsonArray branches)
        {
            foreach (var branch in branches)
                Walk(root, branch!, path, keys);
        }
        if (node["properties"] is JsonObject properties)
        {
            foreach (var (key, value) in properties)
            {
                keys.Add($"{path}.{key}");
                Walk(root, value!, $"{path}.{key}", keys);
            }
        }
    }

    /// <summary>Every object key path in written JSON, arrays of objects folded to <c>[]</c>.</summary>
    private static IEnumerable<string> Keys(JsonNode? node, string path = "$")
    {
        switch (node)
        {
            case JsonObject o:
                foreach (var (key, value) in o)
                {
                    yield return $"{path}.{key}";
                    foreach (var k in Keys(value, $"{path}.{key}"))
                        yield return k;
                }
                break;
            case JsonArray a:
                foreach (var item in a)
                {
                    foreach (var k in Keys(item, $"{path}[]"))
                        yield return k;
                }
                break;
        }
    }

    private static void Compare(HashSet<string> schema, HashSet<string> written)
    {
        var unknown = written.Except(schema).Order().ToList();
        Assert.True(unknown.Count == 0, $"the client writes keys the engine does not have: {string.Join(", ", unknown)}");
        var inexpressible = schema.Except(written).Except(NotNeeded).Order().ToList();
        Assert.True(inexpressible.Count == 0, $"the engine reads keys the client cannot write: {string.Join(", ", inexpressible)}");
    }

    private static string[] Literals(JsonObject root, string definition, string property)
        => [.. root["$defs"]![definition]!["properties"]![property]!["enum"]!.AsArray().Select(v => v!.GetValue<string>())];

    /// <summary>Where two JSON trees differ, or null when they are the same.</summary>
    private static string? Difference(JsonNode? expected, JsonNode? actual, string path)
    {
        switch (expected, actual)
        {
            case (null, null):
                return null;
            case (null, _) or (_, null):
                return $"{path}: {expected?.ToJsonString() ?? "null"} vs {actual?.ToJsonString() ?? "null"}";
            case (JsonObject e, JsonObject a):
            {
                var missing = e.Select(p => p.Key).Except(a.Select(p => p.Key)).ToList();
                var extra = a.Select(p => p.Key).Except(e.Select(p => p.Key)).ToList();
                if (missing.Count > 0 || extra.Count > 0)
                    return $"{path}: record lacks [{string.Join(", ", missing)}], adds [{string.Join(", ", extra)}]";
                foreach (var (key, value) in e)
                {
                    if (Difference(value, a[key], $"{path}.{key}") is { } d)
                        return d;
                }
                return null;
            }
            case (JsonArray e, JsonArray a):
            {
                if (e.Count != a.Count)
                    return $"{path}: {e.Count} elements vs {a.Count}";
                for (var i = 0; i < e.Count; i++)
                {
                    if (Difference(e[i], a[i], $"{path}[{i}]") is { } d)
                        return d;
                }
                return null;
            }
            case (JsonValue e, JsonValue a) when e.GetValueKind() == JsonValueKind.Number && a.GetValueKind() == JsonValueKind.Number:
                return e.GetValue<double>() == a.GetValue<double>() ? null : $"{path}: {e} vs {a}";
            default:
                return JsonNode.DeepEquals(expected, actual) ? null : $"{path}: {expected.ToJsonString()} vs {actual.ToJsonString()}";
        }
    }
}
