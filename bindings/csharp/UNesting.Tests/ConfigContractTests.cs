using System.Text.Json;
using UNesting.Models;
using Xunit;

namespace UNesting.Tests;

/// <summary>
/// Wire-contract guards for the C# request/response DTOs.
///
/// The Rust FFI/WASM layer deserializes requests into <c>ConfigRequest</c>,
/// <c>Request2D</c>, etc., all annotated <c>#[serde(deny_unknown_fields)]</c> — any
/// JSON key the C# side emits that the Rust struct does not declare causes the
/// <b>entire request to be rejected</b>. These tests pin every emitted config key to
/// the canonical Rust wire names so a drift (e.g. C# emitting "generations" while
/// Rust expects "max_generations") fails here instead of at a consumer's runtime.
/// </summary>
public class ConfigContractTests
{
    // Canonical config keys accepted by Rust `ConfigRequest`
    // (crates/core/src/api_types.rs — keep in sync; this set is the tripwire).
    private static readonly HashSet<string> CanonicalConfigKeys = new()
    {
        "strategy",
        "spacing",
        "margin",
        "time_limit_ms",
        "target_utilization",
        "population_size",
        "max_generations",
        "crossover_rate",
        "mutation_rate",
        "seed",
        "multi_sheet",
    };

    private static IEnumerable<string> SerializedKeys(object dto)
    {
        var json = JsonSerializer.Serialize(dto);
        using var doc = JsonDocument.Parse(json);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            yield return prop.Name;
        }
    }

    [Fact]
    public void Config2D_emits_only_canonical_wire_keys()
    {
        // Every field non-default so JsonIgnore(WhenWritingDefault) keeps them all.
        var config = new Config2D
        {
            Strategy = "brkga",
            Spacing = 2.0,
            Margin = 5.0,
            TimeLimitMs = 30000,
            PopulationSize = 100,
            MaxGenerations = 200,
            MultiSheet = true,
        };

        var keys = SerializedKeys(config).ToHashSet();

        Assert.Subset(CanonicalConfigKeys, keys);
        // Regression for the generations/max_generations drift specifically.
        Assert.Contains("max_generations", keys);
        Assert.DoesNotContain("generations", keys);
        Assert.Contains("multi_sheet", keys);
    }

    [Fact]
    public void Config3D_emits_only_canonical_wire_keys()
    {
        var config = new Config3D
        {
            Strategy = "brkga",
            TimeLimitMs = 30000,
            PopulationSize = 100,
            MaxGenerations = 200,
        };

        var keys = SerializedKeys(config).ToHashSet();

        Assert.Subset(CanonicalConfigKeys, keys);
        Assert.Contains("max_generations", keys);
        Assert.DoesNotContain("generations", keys);
    }

    [Fact]
    public void Config2D_default_omits_optional_keys()
    {
        // A default config must not emit zero/false optionals (JsonIgnore), so the
        // single-sheet default path is not accidentally forced into multi_sheet etc.
        var keys = SerializedKeys(new Config2D()).ToHashSet();

        Assert.DoesNotContain("multi_sheet", keys);
        Assert.DoesNotContain("max_generations", keys);
        Assert.Subset(CanonicalConfigKeys, keys);
    }

    [Fact]
    public void NestingResult_deserializes_canonical_response()
    {
        // A canonical 2D wire response (multi-sheet) must round-trip into the DTO,
        // including total_requested and per-placement sheet_index.
        const string wire = """
        {
          "version": "1",
          "success": true,
          "error": null,
          "placements": [
            { "id": "part", "instance": 0, "x": 10.0, "y": 20.0, "rotation": 0.0, "sheet_index": 0, "flipped": false },
            { "id": "part", "instance": 9, "x": 10.0, "y": 20.0, "rotation": 0.0, "sheet_index": 2, "flipped": false }
          ],
          "sheets_used": 3,
          "utilization": 0.71,
          "total_requested": 20,
          "unplaced": ["part"],
          "unplaced_count": 18,
          "all_placed": false,
          "used_bounding_box": [120.0, 340.0],
          "used_utilization": 0.83,
          "elapsed_ms": 5
        }
        """;

        var result = JsonSerializer.Deserialize<NestingResult>(wire);

        Assert.NotNull(result);
        Assert.True(result!.Success);
        Assert.Equal(3, result.SheetsUsed);
        Assert.Equal(20, result.TotalRequested);
        Assert.Equal(2, result.Placements.Count);
        Assert.Equal(2, result.Placements[1].SheetIndex);
        // Bucket B accounting: instance-level unplaced count + all_placed flag,
        // and the padding-independent used-footprint metrics.
        Assert.Equal(18, result.UnplacedCount);
        Assert.False(result.AllPlaced);
        Assert.Equal(new[] { "part" }, result.Unplaced);
        Assert.Equal(120.0, result.UsedBoundingBox[0]);
        Assert.Equal(340.0, result.UsedBoundingBox[1]);
        Assert.Equal(0.83, result.UsedUtilization);
    }
}
