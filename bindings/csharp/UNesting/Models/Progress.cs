using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>
/// Progress information received during solving.
/// </summary>
public class ProgressInfo
{
    /// <summary>
    /// Current iteration number.
    /// </summary>
    [JsonPropertyName("iteration")]
    public int Iteration { get; set; }

    /// <summary>
    /// Total number of iterations (if known).
    /// </summary>
    [JsonPropertyName("total_iterations")]
    public int TotalIterations { get; set; }

    /// <summary>
    /// Current material/volume utilization.
    /// </summary>
    [JsonPropertyName("utilization")]
    public double Utilization { get; set; }

    /// <summary>
    /// Best fitness value found so far.
    /// </summary>
    [JsonPropertyName("best_fitness")]
    public double BestFitness { get; set; }

    /// <summary>
    /// Number of items placed.
    /// </summary>
    [JsonPropertyName("items_placed")]
    public int ItemsPlaced { get; set; }

    /// <summary>
    /// Total number of items.
    /// </summary>
    [JsonPropertyName("total_items")]
    public int TotalItems { get; set; }

    /// <summary>
    /// Elapsed time in milliseconds.
    /// </summary>
    [JsonPropertyName("elapsed_ms")]
    public long ElapsedMs { get; set; }

    /// <summary>
    /// Current solving phase.
    /// </summary>
    [JsonPropertyName("phase")]
    public string Phase { get; set; } = string.Empty;

    /// <summary>
    /// Whether the solver is still running.
    /// </summary>
    [JsonPropertyName("running")]
    public bool Running { get; set; }
}

/// <summary>
/// Event arguments for progress events.
/// </summary>
public class ProgressEventArgs : EventArgs
{
    /// <summary>
    /// Progress information.
    /// </summary>
    public ProgressInfo Progress { get; }

    /// <summary>
    /// Set to true to cancel the operation.
    /// </summary>
    public bool Cancel { get; set; }

    /// <summary>
    /// Creates a new ProgressEventArgs instance.
    /// </summary>
    public ProgressEventArgs(ProgressInfo progress)
    {
        Progress = progress;
    }
}
