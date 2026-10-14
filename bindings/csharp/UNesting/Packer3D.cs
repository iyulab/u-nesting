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
            // A token that can be cancelled needs the progress path: the
            // engine hears a cancel only through its progress callback.
            if (progress == null)
                return cancellationToken.CanBeCanceled
                    ? SolveWithProgress(request, cancellationToken)
                    : Solve(request);

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

    /// <summary>
    /// Checks a packing against the problem it answers — every box inside the container and its
    /// margin, no two boxes overlapping or closer than the spacing, each in an orientation its
    /// constraint allows, one container, each copy packed once, the mass limit, support under
    /// gravity or stability, and <see cref="PackingResult.Unplaced"/>. The request is read as
    /// <see cref="Solve"/> reads it. A packing that breaks the problem is a result, not an
    /// exception.
    /// </summary>
    /// <param name="request">The problem, as given to <see cref="Solve"/>.</param>
    /// <param name="result">The packing: a solve's result, or one built elsewhere in its shape.</param>
    /// <exception cref="NestingException">The engine refused the request.</exception>
    public LayoutCheck Check(PackingRequest request, PackingResult result)
    {
        ThrowIfDisposed();
        request.CheckFinite();
        var solved = JsonSerializer.SerializeToNode(result, NestingJson.Default.PackingResult)!.AsObject();
        solved["success"] = true;
        solved["error"] = null;
        solved["code"] = null;
        solved["details"] = null;
        var body = new System.Text.Json.Nodes.JsonObject
        {
            ["problem"] = JsonSerializer.SerializeToNode(request, NestingJson.Default.PackingRequest),
            ["solve_result"] = solved,
        };
        return NativeCall.Run(NativeLibrary.unesting_check_3d, body.ToJsonString(), NestingJson.Default.LayoutCheck, Observers);
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
