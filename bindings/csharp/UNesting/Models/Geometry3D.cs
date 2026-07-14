using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>
/// Represents a 3D box geometry for bin packing.
/// </summary>
public class Geometry3D
{
    /// <summary>
    /// Unique identifier for this geometry.
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Dimensions as [width, height, depth].
    /// </summary>
    [JsonPropertyName("dimensions")]
    public double[] Dimensions { get; set; } = new double[3];

    /// <summary>
    /// Number of copies to place.
    /// </summary>
    [JsonPropertyName("quantity")]
    public int Quantity { get; set; } = 1;

    /// <summary>
    /// Mass of the item (for stability calculations).
    /// </summary>
    [JsonPropertyName("mass")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public double Mass { get; set; }

    /// <summary>
    /// Whether the item is fragile (affects stacking).
    /// </summary>
    [JsonPropertyName("fragile")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Fragile { get; set; }

    /// <summary>
    /// Maximum items that can be stacked on top.
    /// </summary>
    [JsonPropertyName("max_stack")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int MaxStack { get; set; }

    /// <summary>
    /// Allowed orientations (e.g., ["xyz", "xzy", "yxz"]).
    /// </summary>
    [JsonPropertyName("orientations")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string[]? Orientations { get; set; }

    /// <summary>
    /// Creates a box geometry with given dimensions.
    /// </summary>
    public static Geometry3D Box(string id, double width, double height, double depth, int quantity = 1)
    {
        return new Geometry3D
        {
            Id = id,
            Dimensions = new[] { width, height, depth },
            Quantity = quantity
        };
    }
}

/// <summary>
/// Represents a 3D bin packing boundary (container).
/// </summary>
public class Boundary3D
{
    /// <summary>
    /// Container dimensions as [width, height, depth].
    /// </summary>
    [JsonPropertyName("dimensions")]
    public double[] Dimensions { get; set; } = new double[3];

    /// <summary>
    /// Maximum allowed mass.
    /// </summary>
    [JsonPropertyName("max_mass")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public double MaxMass { get; set; }

    /// <summary>
    /// Enable gravity simulation.
    /// </summary>
    [JsonPropertyName("gravity")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Gravity { get; set; }

    /// <summary>
    /// Enable stability checks.
    /// </summary>
    [JsonPropertyName("stability")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Stability { get; set; }
}

/// <summary>
/// Configuration options for 3D bin packing.
/// </summary>
public class Config3D
{
    /// <summary>
    /// Packing strategy: "blf", "ep", "ga", "brkga", "sa".
    /// </summary>
    [JsonPropertyName("strategy")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Strategy { get; set; }

    /// <summary>
    /// Time limit in milliseconds.
    /// </summary>
    [JsonPropertyName("time_limit_ms")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int TimeLimitMs { get; set; }

    /// <summary>
    /// Population size for GA/BRKGA strategies.
    /// </summary>
    [JsonPropertyName("population_size")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int PopulationSize { get; set; }

    /// <summary>
    /// Maximum number of generations for GA/BRKGA strategies.
    /// </summary>
    [JsonPropertyName("max_generations")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int MaxGenerations { get; set; }
}

/// <summary>
/// Request payload for 3D bin packing.
/// </summary>
public class PackingRequest
{
    /// <summary>
    /// Mode indicator (always "3d" for packing).
    /// </summary>
    [JsonPropertyName("mode")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Mode { get; set; }

    /// <summary>
    /// List of geometries to pack.
    /// </summary>
    [JsonPropertyName("geometries")]
    public List<Geometry3D> Geometries { get; set; } = new();

    /// <summary>
    /// Boundary (container) to pack into.
    /// </summary>
    [JsonPropertyName("boundary")]
    public Boundary3D Boundary { get; set; } = new();

    /// <summary>
    /// Configuration options.
    /// </summary>
    [JsonPropertyName("config")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Config3D? Config { get; set; }
}

/// <summary>
/// A single placement in the packing result.
/// </summary>
public class Placement3D
{
    /// <summary>
    /// Geometry ID.
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Bin index (0-based).
    /// </summary>
    [JsonPropertyName("bin_index")]
    public int BinIndex { get; set; }

    /// <summary>
    /// X position.
    /// </summary>
    [JsonPropertyName("x")]
    public double X { get; set; }

    /// <summary>
    /// Y position.
    /// </summary>
    [JsonPropertyName("y")]
    public double Y { get; set; }

    /// <summary>
    /// Z position.
    /// </summary>
    [JsonPropertyName("z")]
    public double Z { get; set; }

    /// <summary>
    /// Orientation (e.g., "xyz", "xzy").
    /// </summary>
    [JsonPropertyName("orientation")]
    public string Orientation { get; set; } = "xyz";
}

/// <summary>
/// Result of a 3D bin packing operation.
/// </summary>
public class PackingResult
{
    /// <summary>
    /// Whether the operation was successful.
    /// </summary>
    [JsonPropertyName("success")]
    public bool Success { get; set; }

    /// <summary>
    /// List of placements.
    /// </summary>
    [JsonPropertyName("placements")]
    public List<Placement3D> Placements { get; set; } = new();

    /// <summary>
    /// Number of bins used.
    /// </summary>
    [JsonPropertyName("bins_used")]
    public int BinsUsed { get; set; }

    /// <summary>
    /// Volume utilization (0.0 to 1.0).
    /// </summary>
    [JsonPropertyName("utilization")]
    public double Utilization { get; set; }

    /// <summary>
    /// Total number of geometry <b>instances</b> requested (sum of every
    /// geometry's quantity). <see cref="Placements"/> is instance-level while
    /// <see cref="Unplaced"/> lists deduplicated geometry IDs, so the
    /// instance-level unplaced count is <c>TotalRequested - Placements.Count</c>.
    /// </summary>
    [JsonPropertyName("total_requested")]
    public int TotalRequested { get; set; }

    /// <summary>
    /// Geometry IDs (deduplicated) with at least one unplaced instance. See
    /// <see cref="UnplacedCount"/> for the instance-level count.
    /// </summary>
    [JsonPropertyName("unplaced")]
    public List<string> Unplaced { get; set; } = new();

    /// <summary>
    /// Instance-level count of geometry instances that could not be placed
    /// (<c>TotalRequested - Placements.Count</c>). Satisfies the invariant
    /// <c>Placements.Count + UnplacedCount == TotalRequested</c>.
    /// </summary>
    [JsonPropertyName("unplaced_count")]
    public int UnplacedCount { get; set; }

    /// <summary>
    /// Whether every requested instance was placed. Prefer this over
    /// <see cref="Success"/> to detect partial packing.
    /// </summary>
    [JsonPropertyName("all_placed")]
    public bool AllPlaced { get; set; }

    /// <summary>
    /// Solving time in milliseconds.
    /// </summary>
    [JsonPropertyName("elapsed_ms")]
    public long ElapsedMs { get; set; }

    /// <summary>
    /// Error message if success is false.
    /// </summary>
    [JsonPropertyName("error")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Error { get; set; }
}
