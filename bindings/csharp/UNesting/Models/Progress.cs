namespace UNesting.Models;

/// <summary>A progress report from a running solve.</summary>
/// <param name="Iteration">Current iteration or generation.</param>
/// <param name="TotalIterations">Iterations expected in all; 0 when not known.</param>
/// <param name="Utilization">Best utilization so far, from 0 to 1.</param>
/// <param name="BestFitness">Best fitness so far.</param>
/// <param name="ItemsPlaced">Copies placed in the best solution so far.</param>
/// <param name="TotalItems">Copies requested.</param>
/// <param name="ElapsedMs">Time since the solve started, in milliseconds.</param>
/// <param name="Phase">What the solver is doing, as readable text.</param>
/// <param name="Running">Whether the solver is still running.</param>
public sealed record ProgressInfo(
    int Iteration,
    int TotalIterations,
    double Utilization,
    double BestFitness,
    int ItemsPlaced,
    int TotalItems,
    long ElapsedMs,
    string Phase,
    bool Running);

/// <summary>A progress report, and the way to stop the solve from the handler.</summary>
public class ProgressEventArgs : EventArgs
{
    /// <summary>The report.</summary>
    public ProgressInfo Progress { get; }

    /// <summary>Set to true to cancel the solve; it then raises <see cref="OperationCanceledException"/>.</summary>
    public bool Cancel { get; set; }

    /// <summary>Creates the arguments for one report.</summary>
    public ProgressEventArgs(ProgressInfo progress)
    {
        Progress = progress;
    }
}
