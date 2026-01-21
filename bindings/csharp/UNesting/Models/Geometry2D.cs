using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>
/// Represents a 2D polygon geometry for nesting.
/// </summary>
public class Geometry2D
{
    /// <summary>
    /// Unique identifier for this geometry.
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Polygon vertices as [[x, y], ...] array.
    /// </summary>
    [JsonPropertyName("polygon")]
    public double[][] Polygon { get; set; } = Array.Empty<double[]>();

    /// <summary>
    /// Number of copies to place.
    /// </summary>
    [JsonPropertyName("quantity")]
    public int Quantity { get; set; } = 1;

    /// <summary>
    /// Allowed rotation angles in degrees.
    /// </summary>
    [JsonPropertyName("rotations")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double[]? Rotations { get; set; }

    /// <summary>
    /// Whether flipping (mirroring) is allowed.
    /// </summary>
    [JsonPropertyName("allow_flip")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool AllowFlip { get; set; }

    /// <summary>
    /// Interior holes as arrays of vertices.
    /// </summary>
    [JsonPropertyName("holes")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double[][][]? Holes { get; set; }

    /// <summary>
    /// Creates a rectangle geometry.
    /// </summary>
    public static Geometry2D Rectangle(string id, double width, double height, int quantity = 1)
    {
        return new Geometry2D
        {
            Id = id,
            Polygon = new[]
            {
                new[] { 0.0, 0.0 },
                new[] { width, 0.0 },
                new[] { width, height },
                new[] { 0.0, height }
            },
            Quantity = quantity
        };
    }
}

/// <summary>
/// Represents a 2D nesting boundary (sheet/bin).
/// </summary>
public class Boundary2D
{
    /// <summary>
    /// Width of the boundary.
    /// </summary>
    [JsonPropertyName("width")]
    public double Width { get; set; }

    /// <summary>
    /// Height of the boundary.
    /// </summary>
    [JsonPropertyName("height")]
    public double Height { get; set; }
}

/// <summary>
/// Configuration options for 2D nesting.
/// </summary>
public class Config2D
{
    /// <summary>
    /// Nesting strategy: "blf", "nfp", "ga", "brkga", "sa".
    /// </summary>
    [JsonPropertyName("strategy")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Strategy { get; set; }

    /// <summary>
    /// Minimum spacing between parts.
    /// </summary>
    [JsonPropertyName("spacing")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public double Spacing { get; set; }

    /// <summary>
    /// Margin from boundary edges.
    /// </summary>
    [JsonPropertyName("margin")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public double Margin { get; set; }

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
    /// Number of generations for GA/BRKGA strategies.
    /// </summary>
    [JsonPropertyName("generations")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int Generations { get; set; }
}

/// <summary>
/// Request payload for 2D nesting.
/// </summary>
public class NestingRequest
{
    /// <summary>
    /// Mode indicator (always "2d" for nesting).
    /// </summary>
    [JsonPropertyName("mode")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Mode { get; set; }

    /// <summary>
    /// List of geometries to nest.
    /// </summary>
    [JsonPropertyName("geometries")]
    public List<Geometry2D> Geometries { get; set; } = new();

    /// <summary>
    /// Boundary (sheet) to nest into.
    /// </summary>
    [JsonPropertyName("boundary")]
    public Boundary2D Boundary { get; set; } = new();

    /// <summary>
    /// Configuration options.
    /// </summary>
    [JsonPropertyName("config")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Config2D? Config { get; set; }
}

/// <summary>
/// A single placement in the nesting result.
/// </summary>
public class Placement2D
{
    /// <summary>
    /// Geometry ID.
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Sheet/bin index (0-based).
    /// </summary>
    [JsonPropertyName("sheet_index")]
    public int SheetIndex { get; set; }

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
    /// Rotation angle in degrees.
    /// </summary>
    [JsonPropertyName("rotation")]
    public double Rotation { get; set; }

    /// <summary>
    /// Whether the part is flipped.
    /// </summary>
    [JsonPropertyName("flipped")]
    public bool Flipped { get; set; }
}

/// <summary>
/// Result of a 2D nesting operation.
/// </summary>
public class NestingResult
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
    public List<Placement2D> Placements { get; set; } = new();

    /// <summary>
    /// Number of sheets used.
    /// </summary>
    [JsonPropertyName("sheets_used")]
    public int SheetsUsed { get; set; }

    /// <summary>
    /// Material utilization (0.0 to 1.0).
    /// </summary>
    [JsonPropertyName("utilization")]
    public double Utilization { get; set; }

    /// <summary>
    /// Items that could not be placed.
    /// </summary>
    [JsonPropertyName("unplaced")]
    public List<string> Unplaced { get; set; } = new();

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
