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
    /// Maximum number of generations for GA/BRKGA strategies.
    /// </summary>
    [JsonPropertyName("max_generations")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int MaxGenerations { get; set; }

    /// <summary>
    /// Optional RNG seed for reproducible stochastic runs (GA, BRKGA, SA). When
    /// null the solver seeds from entropy. Reproducibility holds only when the
    /// generation cap (not the wall-clock time limit) terminates the run.
    /// </summary>
    [JsonPropertyName("seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? Seed { get; set; }

    /// <summary>
    /// Distribute overflow across multiple sheets. When true, parts that do not fit
    /// on a single sheet spill onto additional sheets instead of becoming unplaced;
    /// <see cref="NestingResult.SheetsUsed"/> reports the sheet count and each
    /// placement's <see cref="Placement2D.SheetIndex"/> selects its sheet (with
    /// sheet-local coordinates). Defaults to false (single-sheet solve).
    /// </summary>
    [JsonPropertyName("multi_sheet")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool MultiSheet { get; set; }
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
    /// Total number of geometry <b>instances</b> requested (sum of every
    /// geometry's quantity). <see cref="Placements"/> is instance-level while
    /// <see cref="Unplaced"/> lists deduplicated geometry IDs, so the
    /// instance-level unplaced count is <c>TotalRequested - Placements.Count</c>.
    /// </summary>
    [JsonPropertyName("total_requested")]
    public int TotalRequested { get; set; }

    /// <summary>
    /// Geometry IDs (deduplicated) with at least one unplaced instance. Tells
    /// <i>which</i> geometries failed; see <see cref="UnplacedCount"/> for <i>how many</i>.
    /// </summary>
    [JsonPropertyName("unplaced")]
    public List<string> Unplaced { get; set; } = new();

    /// <summary>
    /// Instance-level count of geometry instances that could not be placed
    /// (<c>TotalRequested - Placements.Count</c>). Satisfies the invariant
    /// <c>Placements.Count + UnplacedCount == TotalRequested</c>; unlike
    /// <see cref="Unplaced"/> it never undercounts a multi-quantity geometry.
    /// </summary>
    [JsonPropertyName("unplaced_count")]
    public int UnplacedCount { get; set; }

    /// <summary>
    /// Whether every requested instance was placed
    /// (<c>Placements.Count == TotalRequested</c>). Prefer this over
    /// <see cref="Success"/> to detect partial packing: <see cref="Success"/>
    /// only means the solve completed without error, not that all pieces fit.
    /// </summary>
    [JsonPropertyName("all_placed")]
    public bool AllPlaced { get; set; }

    /// <summary>
    /// Axis-aligned bounding box <c>[width, height]</c> of the placed pieces'
    /// actual footprint. Boundary-padding independent, unlike
    /// <see cref="Utilization"/> (which divides by the full boundary and shrinks
    /// as boundary height grows). For an open-ended roll the larger axis is the
    /// material length consumed.
    /// </summary>
    [JsonPropertyName("used_bounding_box")]
    public double[] UsedBoundingBox { get; set; } = new double[2];

    /// <summary>
    /// Packing-density metric: utilization against the used bounding box
    /// (<c>placed_area / (used_width * used_height)</c>) rather than the full
    /// boundary. The denominator shrinks on <b>both</b> axes to the placed
    /// footprint, so this measures how tightly pieces are packed within their
    /// own extent, independent of boundary padding.
    /// <para>
    /// This is <b>not</b> a fixed-width stock-efficiency metric. For fixed-width
    /// material (fabric rolls, coil, sheet stock) the boundary width is real
    /// consumed stock, not padding — collapsing the width axis over-reports
    /// material savings. Fixed-width consumers should compute
    /// <c>placed_area / (boundary_width * UsedBoundingBox[1])</c> instead.
    /// </para>
    /// </summary>
    [JsonPropertyName("used_utilization")]
    public double UsedUtilization { get; set; }

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
