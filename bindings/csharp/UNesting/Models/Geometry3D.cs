using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>A 3D packing strategy, by the name the engine reads.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<Strategy3D>))]
public enum Strategy3D
{
    /// <summary>Layer-by-layer bottom-left fill: fast.</summary>
    [JsonStringEnumMemberName("blf")] BottomLeftFill,
    /// <summary>Extreme-point heuristic.</summary>
    [JsonStringEnumMemberName("ep")] ExtremePoint,
    /// <summary>Genetic algorithm over packing order and orientation.</summary>
    [JsonStringEnumMemberName("ga")] GeneticAlgorithm,
    /// <summary>Biased random-key genetic algorithm.</summary>
    [JsonStringEnumMemberName("brkga")] Brkga,
    /// <summary>Simulated annealing.</summary>
    [JsonStringEnumMemberName("sa")] SimulatedAnnealing,
}

/// <summary>Which turns a box may be packed in.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<OrientationConstraint>))]
public enum OrientationConstraint
{
    /// <summary>Any of the six axis-aligned orientations.</summary>
    [JsonStringEnumMemberName("any")] Any,
    /// <summary>Height stays vertical; the box may turn a quarter about it (width and depth swap).</summary>
    [JsonStringEnumMemberName("upright")] Upright,
    /// <summary>As given: width along x, depth along y, height along z.</summary>
    [JsonStringEnumMemberName("fixed")] Fixed,
}

/// <summary>
/// How a placed box is turned: which of its own axes — <c>x</c> its width, <c>y</c> its
/// depth, <c>z</c> its height — lies along the container's x, y and z, in that order.
/// <see cref="Yxz"/> is a quarter turn about the vertical: the box's depth along x, its
/// width along y.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<AxisPermutation>))]
public enum AxisPermutation
{
    /// <summary>As given.</summary>
    [JsonStringEnumMemberName("xyz")] Xyz,
    /// <summary>Width along x, height along y, depth along z.</summary>
    [JsonStringEnumMemberName("xzy")] Xzy,
    /// <summary>Depth along x, width along y, height along z.</summary>
    [JsonStringEnumMemberName("yxz")] Yxz,
    /// <summary>Depth along x, height along y, width along z.</summary>
    [JsonStringEnumMemberName("yzx")] Yzx,
    /// <summary>Height along x, width along y, depth along z.</summary>
    [JsonStringEnumMemberName("zxy")] Zxy,
    /// <summary>Height along x, depth along y, width along z.</summary>
    [JsonStringEnumMemberName("zyx")] Zyx,
}

/// <summary>A box to pack, and how many copies of it.</summary>
public class Geometry3D
{
    /// <summary>Identifier, unique within the request; placements carry it back.</summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>Extent along x: above 0.</summary>
    [JsonIgnore]
    public double Width { get; set; }

    /// <summary>Extent along y: above 0.</summary>
    [JsonIgnore]
    public double Depth { get; set; }

    /// <summary>Extent along z, the vertical: above 0.</summary>
    [JsonIgnore]
    public double Height { get; set; }

    /// <summary>The wire form of the size: <c>[width, depth, height]</c>, the engine's order.</summary>
    public double[] Dimensions => [Width, Depth, Height];

    /// <summary>Number of copies to place: at least 1.</summary>
    public int Quantity { get; set; } = 1;

    /// <summary>Mass of one copy, at least 0; counted against <see cref="Boundary3D.MaxMass"/>.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Mass { get; set; }

    /// <summary>The turns allowed; <see cref="OrientationConstraint.Any"/> when null.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public OrientationConstraint? Orientation { get; set; }

    /// <summary>A <paramref name="width"/> × <paramref name="depth"/> × <paramref name="height"/> box.</summary>
    public static Geometry3D Box(string id, double width, double depth, double height, int quantity = 1) => new()
    {
        Id = id,
        Width = width,
        Depth = depth,
        Height = height,
        Quantity = quantity,
    };
}

/// <summary>The container to pack into.</summary>
public class Boundary3D
{
    /// <summary>Extent along x: above 0.</summary>
    [JsonIgnore]
    public double Width { get; set; }

    /// <summary>Extent along y: above 0.</summary>
    [JsonIgnore]
    public double Depth { get; set; }

    /// <summary>Extent along z, the vertical: above 0.</summary>
    [JsonIgnore]
    public double Height { get; set; }

    /// <summary>The wire form of the size: <c>[width, depth, height]</c>, the engine's order.</summary>
    public double[] Dimensions => [Width, Depth, Height];

    /// <summary>Total mass the container may hold, above 0; unlimited when null.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? MaxMass { get; set; }

    /// <summary>Whether every box must rest on the floor or on boxes below it.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Gravity { get; set; }

    /// <summary>Whether stacks are checked for stability.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Stability { get; set; }

    /// <summary>A <paramref name="width"/> × <paramref name="depth"/> × <paramref name="height"/> container.</summary>
    public static Boundary3D Box(double width, double depth, double height) => new()
    {
        Width = width,
        Depth = depth,
        Height = height,
    };
}

/// <summary>
/// Settings for 3D packing. A <c>null</c> setting is left out of the request, so the
/// engine's default applies.
/// </summary>
public class Config3D
{
    /// <summary>The strategy; <see cref="Strategy3D.BottomLeftFill"/> when null.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Strategy3D? Strategy { get; set; }

    /// <summary>Minimum distance between any two packed boxes, at least 0 (0 when null).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Spacing { get; set; }

    /// <summary>Minimum distance between any packed box and the container walls, at least 0 (0 when null).</summary>
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
    /// value given. When null each strategy uses the size it is tuned for: 100 for GA, 50
    /// for BRKGA.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? PopulationSize { get; set; }

    /// <summary>
    /// Generation budget for the GA/BRKGA strategies (0 evaluates the first population
    /// only); the search runs with the value given. When null each strategy uses the
    /// budget it is tuned for: 500 for GA, 100 for BRKGA.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? MaxGenerations { get; set; }

    /// <summary>
    /// Iteration budget for the single-solution searches, at least 1; null runs each with the
    /// budget it is tuned for: 10000 for SA. A seeded run that ends on this budget rather than on
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

    /// <summary>RNG seed for the stochastic strategies (GA, BRKGA, SA).</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ulong? Seed { get; set; }
}

/// <summary>A 3D packing problem: the boxes, the container and the settings.</summary>
public class PackingRequest
{
    /// <summary>The boxes to pack: at least one (an empty list is refused, <c>empty_input</c>).</summary>
    public List<Geometry3D> Geometries { get; set; } = [];

    /// <summary>The container to pack into.</summary>
    public Boundary3D Boundary { get; set; } = new();

    /// <summary>Settings, or <c>null</c> for the engine's defaults.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Config3D? Config { get; set; }

    /// <summary>
    /// Refuses a NaN or infinite number — JSON has neither, so it cannot reach the engine —
    /// with the engine's <c>value_not_finite</c>, naming where it sits.
    /// </summary>
    internal void CheckFinite()
    {
        for (var i = 0; i < Geometries.Count; i++)
        {
            var g = Geometries[i];
            Finite.Values($"geometries[{i}].dimensions", g.Dimensions, g.Id);
            Finite.Value($"geometries[{i}].mass", g.Mass, g.Id);
        }
        Finite.Values("boundary.dimensions", Boundary.Dimensions);
        Finite.Value("boundary.max_mass", Boundary.MaxMass);
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

/// <summary>Where one copy of a box was packed.</summary>
/// <param name="Id">The box's <see cref="Geometry3D.Id"/>.</param>
/// <param name="Instance">Which copy of the box, from 0.</param>
/// <param name="BinIndex">Container the copy is in, from 0.</param>
/// <param name="X">Position along x (width).</param>
/// <param name="Y">Position along y (depth).</param>
/// <param name="Z">Position along z (height).</param>
/// <param name="Orientation">How the copy is turned.</param>
public sealed record Placement3D(
    string Id,
    int Instance,
    int BinIndex,
    double X,
    double Y,
    double Z,
    AxisPermutation Orientation);

/// <summary>A solved 3D packing problem. A request the engine refuses raises <see cref="NestingException"/> instead.</summary>
/// <param name="Version">The engine's version.</param>
/// <param name="Placements">One per packed copy.</param>
/// <param name="BinsUsed">Number of containers used.</param>
/// <param name="Utilization">Volume utilization, from 0 to 1.</param>
/// <param name="TotalRequested">
/// Copies requested (the sum of every box's quantity). <see cref="Placements"/> is per
/// copy while <see cref="Unplaced"/> lists box ids once each.
/// </param>
/// <param name="Unplaced">Ids of the boxes with at least one copy left unpacked, each once.</param>
/// <param name="UnplacedCount">Copies left unpacked: <c>Placements.Count + UnplacedCount == TotalRequested</c>.</param>
/// <param name="AllPlaced">Whether every requested copy was packed; check this to detect a partial result.</param>
/// <param name="StopReason">
/// Why the run stopped — at the time limit, at the target, on cancellation or at its own end.
/// </param>
/// <param name="ElapsedMs">Solve time in milliseconds.</param>
public sealed record PackingResult(
    string Version,
    IReadOnlyList<Placement3D> Placements,
    int BinsUsed,
    double Utilization,
    int TotalRequested,
    IReadOnlyList<string> Unplaced,
    int UnplacedCount,
    bool AllPlaced,
    StopReason StopReason,
    long ElapsedMs);
