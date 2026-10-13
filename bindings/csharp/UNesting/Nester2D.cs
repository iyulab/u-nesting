using System.Text.Json;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// 2D nesting solver for polygon placement optimization.
/// </summary>
public class Nester2D : IDisposable
{
    private bool _disposed;

    /// <summary>Sees each raw response and progress body before it is read — the contract tests compare them with the records.</summary>
    internal NativeCall.Observers Observers { get; } = new();

    /// <summary>
    /// Event raised when progress is reported during solving.
    /// </summary>
    public event EventHandler<ProgressEventArgs>? ProgressChanged;

    /// <summary>
    /// Solves a 2D nesting problem.
    /// </summary>
    /// <param name="request">The nesting request.</param>
    /// <returns>The nesting result.</returns>
    /// <exception cref="NestingException">The engine refused the request, or a number in it is not finite (<c>value_not_finite</c>).</exception>
    public NestingResult Solve(NestingRequest request)
    {
        ThrowIfDisposed();
        return NativeCall.Run(NativeLibrary.unesting_solve_2d, Write(request), NestingJson.Default.NestingResult, Observers);
    }

    /// <summary>
    /// Solves a 2D nesting problem, raising <see cref="ProgressChanged"/> as the solver reports.
    /// An exception thrown by a handler stops the solve and is rethrown here.
    /// </summary>
    /// <param name="request">The nesting request.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The nesting result.</returns>
    /// <exception cref="NestingException">The engine refused the request, or a number in it is not finite (<c>value_not_finite</c>).</exception>
    /// <exception cref="OperationCanceledException">The solve was cancelled.</exception>
    public NestingResult SolveWithProgress(NestingRequest request, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return NativeCall.Run(
            NativeLibrary.unesting_solve_2d_with_progress, Write(request), NestingJson.Default.NestingResult, Observers,
            Report, cancellationToken);
    }

    /// <summary>
    /// Solves a 2D nesting problem on the thread pool, with progress reporting.
    /// </summary>
    /// <param name="request">The nesting request.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The nesting result.</returns>
    public Task<NestingResult> SolveAsync(
        NestingRequest request,
        IProgress<ProgressInfo>? progress = null,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            if (progress == null)
                return Solve(request);

            void Handler(object? sender, ProgressEventArgs e) => progress.Report(e.Progress);
            ProgressChanged += Handler;
            try
            {
                return SolveWithProgress(request, cancellationToken);
            }
            finally
            {
                ProgressChanged -= Handler;
            }
        }, cancellationToken);
    }

    private static string Write(NestingRequest request)
    {
        request.CheckFinite();
        return JsonSerializer.Serialize(request, NestingJson.Default.NestingRequest);
    }

    private bool Report(ProgressInfo progress)
    {
        var args = new ProgressEventArgs(progress);
        ProgressChanged?.Invoke(this, args);
        return args.Cancel;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Disposes of this instance.
    /// </summary>
    public void Dispose()
    {
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
