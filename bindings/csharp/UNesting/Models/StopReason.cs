using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>
/// Why a solve stopped — the run's own account of the layout it returned. A layout cut
/// short by the time limit and one the method finished look alike; only this says which.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<StopReason>))]
public enum StopReason
{
    /// <summary>The method ran to its own end: a fill tried every part, or a search spent its internal budget.</summary>
    [JsonStringEnumMemberName("completed")] Completed,
    /// <summary>An exact method proved the layout optimal.</summary>
    [JsonStringEnumMemberName("optimal")] Optimal,
    /// <summary>Every part was placed at the target utilization or more, so the search stopped there.</summary>
    [JsonStringEnumMemberName("target_reached")] TargetReached,
    /// <summary>The search spent its generation or iteration budget.</summary>
    [JsonStringEnumMemberName("iteration_limit")] IterationLimit,
    /// <summary>The search went its stall budget of generations without improving, and stopped.</summary>
    [JsonStringEnumMemberName("no_improvement")] NoImprovement,
    /// <summary>The time limit ran out; the layout is the best found by then.</summary>
    [JsonStringEnumMemberName("time_limit")] TimeLimit,
    /// <summary>The caller cancelled the run; the layout is what was placed by then.</summary>
    [JsonStringEnumMemberName("cancelled")] Cancelled,
}
