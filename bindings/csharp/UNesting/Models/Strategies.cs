using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>A setting of a request's configuration, by the key the engine reads.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<NestingSetting>))]
public enum NestingSetting
{
    /// <summary><see cref="Config2D.Spacing"/> / <see cref="Config3D.Spacing"/>.</summary>
    [JsonStringEnumMemberName("spacing")] Spacing,
    /// <summary><see cref="Config2D.Margin"/> / <see cref="Config3D.Margin"/>.</summary>
    [JsonStringEnumMemberName("margin")] Margin,
    /// <summary><see cref="Config2D.TimeLimitMs"/> / <see cref="Config3D.TimeLimitMs"/>.</summary>
    [JsonStringEnumMemberName("time_limit_ms")] TimeLimitMs,
    /// <summary><see cref="Config2D.TargetUtilization"/> / <see cref="Config3D.TargetUtilization"/>.</summary>
    [JsonStringEnumMemberName("target_utilization")] TargetUtilization,
    /// <summary><see cref="Config2D.PopulationSize"/> / <see cref="Config3D.PopulationSize"/>.</summary>
    [JsonStringEnumMemberName("population_size")] PopulationSize,
    /// <summary><see cref="Config2D.MaxGenerations"/> / <see cref="Config3D.MaxGenerations"/>.</summary>
    [JsonStringEnumMemberName("max_generations")] MaxGenerations,
    /// <summary><see cref="Config2D.CrossoverRate"/> / <see cref="Config3D.CrossoverRate"/>.</summary>
    [JsonStringEnumMemberName("crossover_rate")] CrossoverRate,
    /// <summary><see cref="Config2D.MutationRate"/> / <see cref="Config3D.MutationRate"/>.</summary>
    [JsonStringEnumMemberName("mutation_rate")] MutationRate,
    /// <summary><see cref="Config2D.Seed"/> / <see cref="Config3D.Seed"/>.</summary>
    [JsonStringEnumMemberName("seed")] Seed,
    /// <summary><see cref="Config2D.MultiSheet"/> (2D only).</summary>
    [JsonStringEnumMemberName("multi_sheet")] MultiSheet,
}

/// <summary>A 2D strategy and the settings it reads.</summary>
/// <param name="Name">The strategy.</param>
/// <param name="Reads">
/// The settings it reads. A request giving any other setting with this strategy is refused
/// (<see cref="NestingException.ErrorCode"/> <c>invalid_option</c>).
/// </param>
public sealed record StrategyInfo2D(Strategy2D Name, IReadOnlyList<NestingSetting> Reads);

/// <summary>A 3D strategy and the settings it reads.</summary>
/// <param name="Name">The strategy.</param>
/// <param name="Reads">
/// The settings it reads. A request giving any other setting with this strategy is refused
/// (<see cref="NestingException.ErrorCode"/> <c>invalid_option</c>).
/// </param>
public sealed record StrategyInfo3D(Strategy3D Name, IReadOnlyList<NestingSetting> Reads);

/// <summary>
/// The strategies each dimension offers in the native library, and the settings each reads —
/// the same table a solve's refusal of a setting the strategy does not read comes from, so a
/// form can show only the settings that change the chosen strategy's run.
/// </summary>
/// <param name="TwoD">The 2D nesting strategies.</param>
/// <param name="ThreeD">The 3D packing strategies.</param>
public sealed record StrategyCatalog(
    [property: JsonPropertyName("2d")] IReadOnlyList<StrategyInfo2D> TwoD,
    [property: JsonPropertyName("3d")] IReadOnlyList<StrategyInfo3D> ThreeD)
{
    /// <summary>Whether <paramref name="strategy"/> reads <paramref name="setting"/> in 2D.</summary>
    public bool Reads(Strategy2D strategy, NestingSetting setting) =>
        TwoD.Any(s => s.Name == strategy && s.Reads.Contains(setting));

    /// <summary>Whether <paramref name="strategy"/> reads <paramref name="setting"/> in 3D.</summary>
    public bool Reads(Strategy3D strategy, NestingSetting setting) =>
        ThreeD.Any(s => s.Name == strategy && s.Reads.Contains(setting));
}
