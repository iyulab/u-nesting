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
    /// Checks a layout against the problem it answers — every piece within the boundary and its
    /// margin, no two pieces on a sheet overlapping or closer than the spacing, each at a rotation
    /// its geometry allows and flipped only where flipping is allowed, each copy placed once, and
    /// the geometries left short reported in <see cref="NestingResult.Unplaced"/>. The request is
    /// read as <see cref="Solve"/> reads it. A layout that breaks the problem is a result, not an
    /// exception.
    /// </summary>
    /// <param name="request">The problem, as given to <see cref="Solve"/>.</param>
    /// <param name="result">The layout: a solve's result, or one built elsewhere in its shape.</param>
    /// <exception cref="NestingException">The engine refused the request.</exception>
    public LayoutCheck Check(NestingRequest request, NestingResult result)
    {
        ThrowIfDisposed();
        request.CheckFinite();
        var solved = JsonSerializer.SerializeToNode(result, NestingJson.Default.NestingResult)!.AsObject();
        solved["success"] = true;
        solved["error"] = null;
        solved["code"] = null;
        solved["details"] = null;
        var body = new System.Text.Json.Nodes.JsonObject
        {
            ["problem"] = JsonSerializer.SerializeToNode(request, NestingJson.Default.NestingRequest),
            ["solve_result"] = solved,
        };
        return NativeCall.Run(NativeLibrary.unesting_check_2d, body.ToJsonString(), NestingJson.Default.LayoutCheck, Observers);
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
