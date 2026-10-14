using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>A 2D nesting strategy, by the name the engine reads.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<Strategy2D>))]
public enum Strategy2D
{
    /// <summary>Bottom-left fill: fast, good for simple shapes.</summary>
    [JsonStringEnumMemberName("blf")] BottomLeftFill,
    /// <summary>No-fit-polygon guided placement.</summary>
    [JsonStringEnumMemberName("nfp")] NfpGuided,
    /// <summary>Genetic algorithm over placement order and rotation.</summary>
    [JsonStringEnumMemberName("ga")] GeneticAlgorithm,
    /// <summary>Biased random-key genetic algorithm.</summary>
    [JsonStringEnumMemberName("brkga")] Brkga,
    /// <summary>Simulated annealing.</summary>
    [JsonStringEnumMemberName("sa")] SimulatedAnnealing,
    /// <summary>Goal-driven ruin and recreate.</summary>
    [JsonStringEnumMemberName("gdrr")] Gdrr,
    /// <summary>Adaptive large neighbourhood search.</summary>
    [JsonStringEnumMemberName("alns")] Alns,
}

/// <summary>A polygon to place, and how many copies of it.</summary>
public class Geometry2D
{
    /// <summary>Identifier, unique within the request; placements carry it back.</summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>Outer boundary vertices as <c>[[x, y], ...]</c>.</summary>
    public double[][] Polygon { get; set; } = [];

    /// <summary>Interior holes, each as <c>[[x, y], ...]</c>.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double[][][]? Holes { get; set; }

    /// <summary>Number of copies to place: at least 1.</summary>
    public int Quantity { get; set; } = 1;

    /// <summary>
    /// Allowed rotation angles in degrees: at least one. Leave it <c>null</c> for a
    /// fixed orientation; an empty array is refused.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double[]? Rotations { get; set; }

    /// <summary>Whether a copy may be placed mirrored.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool AllowFlip { get; set; }

    /// <summary>A <paramref name="width"/> × <paramref name="height"/> rectangle with its corner at the origin.</summary>
    public static Geometry2D Rectangle(string id, double width, double height, int quantity = 1) => new()
    {
        Id = id,
        Polygon = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        Quantity = quantity,
    };
}

/// <summary>
/// The sheet to nest into: a <see cref="Width"/> × <see cref="Height"/> rectangle, or an
/// arbitrary <see cref="Polygon"/> — exactly one of the two; both or neither is refused
/// (<c>invalid_boundary</c>).
/// </summary>
public class Boundary2D
{
    /// <summary>Width of a rectangular sheet: above 0.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Width { get; set; }

    /// <summary>Height of a rectangular sheet: above 0.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Height { get; set; }

    /// <summary>Vertices of an arbitrary sheet as <c>[[x, y], ...]</c>, at least three.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double[][]? Polygon { get; set; }

    /// <summary>A <paramref name="width"/> × <paramref name="height"/> rectangular sheet.</summary>
    public static Boundary2D Rectangle(double width, double height) => new() { Width = width, Height = height };
}

/// <summary>
/// Settings for 2D nesting. A <c>null</c> setting is left out of the request, so the
/// engine's default applies.
/// </summary>
public class Config2D
{
    /// <summary>The strategy; <see cref="Strategy2D.BottomLeftFill"/> when null.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Strategy2D? Strategy { get; set; }

    /// <summary>
    /// Minimum distance between any two placed parts, at least 0 (0 when null). It does
    /// not apply to the sheet edge; that is <see cref="Margin"/>.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Spacing { get; set; }

    /// <summary>Minimum distance between any placed part and the sheet edge, at least 0 (0 when null).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Margin { get; set; }

    /// <summary>
    /// Time limit in milliseconds; 0 runs without one. When null the engine's default,
    /// 30000, applies.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? TimeLimitMs { get; set; }

    /// <summary>Utilization from 0 to 1 at which the search may stop early.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? TargetUtilization { get; set; }

    /// <summary>
    /// Population size for the GA/BRKGA strategies, at least 2; the search runs with the
    /// value given. When null each strategy uses the size it is tuned for: 30.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? PopulationSize { get; set; }

    /// <summary>
    /// Generation budget for the GA/BRKGA strategies (0 evaluates the first population
    /// only); the search runs with the value given. When null each strategy uses the
    /// budget it is tuned for: 50.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxGenerations { get; set; }

    /// <summary>
    /// Iteration budget for the single-solution searches, at least 1; null runs each with the
    /// budget it is tuned for: 500 for SA, 1000 for GDRR and ALNS. A seeded run that ends on this budget rather than on
    /// the time limit is reproduced exactly.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? MaxIterations { get; set; }

    /// <summary>GA crossover rate from 0 to 1 (0.85 when null).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? CrossoverRate { get; set; }

    /// <summary>GA mutation rate from 0 to 1 (0.05 when null).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? MutationRate { get; set; }

    /// <summary>
    /// RNG seed for reproducible stochastic runs (GA, BRKGA, SA). When null the solver
    /// seeds from entropy. Reproducibility holds only when the generation cap (not the
    /// wall-clock time limit) terminates the run.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? Seed { get; set; }

    /// <summary>
    /// Distribute overflow across multiple sheets. When true, parts that do not fit on a
    /// single sheet spill onto additional sheets instead of becoming unplaced;
    /// <see cref="NestingResult.SheetsUsed"/> reports the sheet count and each placement's
    /// <see cref="Placement2D.SheetIndex"/> selects its sheet (with sheet-local
    /// coordinates). Defaults to false (single-sheet solve).
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool MultiSheet { get; set; }
}

/// <summary>A 2D nesting problem: the parts, the sheet and the settings.</summary>
public class NestingRequest
{
    /// <summary>The parts to place: at least one (an empty list is refused, <c>empty_input</c>).</summary>
    public List<Geometry2D> Geometries { get; set; } = [];

    /// <summary>The sheet to nest into.</summary>
    public Boundary2D Boundary { get; set; } = new();

    /// <summary>Settings, or <c>null</c> for the engine's defaults.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Config2D? Config { get; set; }

    /// <summary>
    /// Refuses a NaN or infinite number — JSON has neither, so it cannot reach the engine —
    /// with the engine's <c>value_not_finite</c>, naming where it sits.
    /// </summary>
    internal void CheckFinite()
    {
        for (var i = 0; i < Geometries.Count; i++)
        {
            var g = Geometries[i];
            var at = $"geometries[{i}]";
            Finite.Points($"{at}.polygon", g.Polygon, g.Id);
            if (g.Holes is { } holes)
            {
                for (var h = 0; h < holes.Length; h++)
                    Finite.Points($"{at}.holes[{h}]", holes[h], g.Id);
            }
            if (g.Rotations is { } rotations)
                Finite.Values($"{at}.rotations", rotations, g.Id);
        }
        Finite.Value("boundary.width", Boundary.Width);
        Finite.Value("boundary.height", Boundary.Height);
        if (Boundary.Polygon is { } polygon)
            Finite.Points("boundary.polygon", polygon);
        if (Config is { } c)
        {
            Finite.Value("config.spacing", c.Spacing);
            Finite.Value("config.margin", c.Margin);
            Finite.Value("config.target_utilization", c.TargetUtilization);
            Finite.Value("config.crossover_rate", c.CrossoverRate);
            Finite.Value("config.mutation_rate", c.MutationRate);
        }
    }
}

/// <summary>Where one copy of a part was placed.</summary>
/// <param name="Id">The part's <see cref="Geometry2D.Id"/>.</param>
/// <param name="Instance">Which copy of the part, from 0.</param>
/// <param name="X">X offset of the placed polygon.</param>
/// <param name="Y">Y offset of the placed polygon.</param>
/// <param name="Rotation">Rotation in degrees, one of the part's allowed angles.</param>
/// <param name="SheetIndex">Sheet the copy is on, from 0; coordinates are local to it.</param>
/// <param name="Flipped">Whether the copy is mirrored.</param>
public sealed record Placement2D(
    string Id,
    int Instance,
    double X,
    double Y,
    double Rotation,
    int SheetIndex,
    bool Flipped);

/// <summary>A solved 2D nesting problem. A request the engine refuses raises <see cref="NestingException"/> instead.</summary>
/// <param name="Version">The engine's version.</param>
/// <param name="Placements">One per placed copy.</param>
/// <param name="SheetsUsed">Number of sheets used.</param>
/// <param name="Utilization">Material utilization of the sheets used, from 0 to 1.</param>
/// <param name="TotalRequested">
/// Copies requested (the sum of every part's quantity). <see cref="Placements"/> is per
/// copy while <see cref="Unplaced"/> lists part ids once each.
/// </param>
/// <param name="Unplaced">Ids of the parts with at least one copy left unplaced, each once; see <see cref="UnplacedCount"/> for how many copies.</param>
/// <param name="UnplacedCount">Copies left unplaced: <c>Placements.Count + UnplacedCount == TotalRequested</c>.</param>
/// <param name="AllPlaced">
/// Whether every requested copy was placed. Check this to detect a partial result: a
/// returned result means the solve completed, not that everything fit.
/// </param>
/// <param name="UsedBoundingBox">
/// <c>[width, height]</c> of the placed parts' footprint, independent of the sheet's
/// padding; for an open-ended roll the larger axis is the material length consumed.
/// <c>[0, 0]</c> in a multi-sheet solve and when nothing was placed.
/// </param>
/// <param name="UsedUtilization">
/// Placed area over <see cref="UsedBoundingBox"/>'s area: how tightly the parts pack
/// within their own extent. Not a stock-efficiency figure for fixed-width material,
/// whose full width is consumed — use <c>placed_area / (sheet_width * UsedBoundingBox[1])</c>
/// there. 0 in a multi-sheet solve and when nothing was placed.
/// </param>
/// <param name="StopReason">
/// Why the run stopped — at the time limit, at the target, on cancellation or at its own end.
/// </param>
/// <param name="ElapsedMs">Solve time in milliseconds.</param>
public sealed record NestingResult(
    string Version,
    IReadOnlyList<Placement2D> Placements,
    int SheetsUsed,
    double Utilization,
    int TotalRequested,
    IReadOnlyList<string> Unplaced,
    int UnplacedCount,
    bool AllPlaced,
    IReadOnlyList<double> UsedBoundingBox,
    double UsedUtilization,
    StopReason StopReason,
    long ElapsedMs);
