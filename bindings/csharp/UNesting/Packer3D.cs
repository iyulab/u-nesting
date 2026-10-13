using System.Text.Json;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// 3D bin packing solver for box placement optimization.
/// </summary>
public class Packer3D : IDisposable
{
    private bool _disposed;

    /// <summary>Sees each raw response and progress body before it is read — the contract tests compare them with the records.</summary>
    internal NativeCall.Observers Observers { get; } = new();

    /// <summary>
    /// Event raised when progress is reported during solving.
    /// </summary>
    public event EventHandler<ProgressEventArgs>? ProgressChanged;

    /// <summary>
    /// Solves a 3D packing problem.
    /// </summary>
    /// <param name="request">The packing request.</param>
    /// <returns>The packing result.</returns>
    /// <exception cref="NestingException">The engine refused the request, or a number in it is not finite (<c>value_not_finite</c>).</exception>
    public PackingResult Solve(PackingRequest request)
    {
        ThrowIfDisposed();
        return NativeCall.Run(NativeLibrary.unesting_solve_3d, Write(request), NestingJson.Default.PackingResult, Observers);
    }

    /// <summary>
    /// Solves a 3D packing problem, raising <see cref="ProgressChanged"/> as the solver reports.
    /// An exception thrown by a handler stops the solve and is rethrown here.
    /// </summary>
    /// <param name="request">The packing request.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The packing result.</returns>
    /// <exception cref="NestingException">The engine refused the request, or a number in it is not finite (<c>value_not_finite</c>).</exception>
    /// <exception cref="OperationCanceledException">The solve was cancelled.</exception>
    public PackingResult SolveWithProgress(PackingRequest request, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return NativeCall.Run(
            NativeLibrary.unesting_solve_3d_with_progress, Write(request), NestingJson.Default.PackingResult, Observers,
            Report, cancellationToken);
    }

    /// <summary>
    /// Solves a 3D packing problem on the thread pool, with progress reporting.
    /// </summary>
    /// <param name="request">The packing request.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The packing result.</returns>
    public Task<PackingResult> SolveAsync(
        PackingRequest request,
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

    private static string Write(PackingRequest request)
    {
        request.CheckFinite();
        return JsonSerializer.Serialize(request, NestingJson.Default.PackingRequest);
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
