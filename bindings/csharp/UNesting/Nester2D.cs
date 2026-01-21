using System.Runtime.InteropServices;
using System.Text.Json;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// 2D nesting solver for polygon placement optimization.
/// </summary>
public class Nester2D : IDisposable
{
    private readonly JsonSerializerOptions _jsonOptions;
    private bool _disposed;

    /// <summary>
    /// Event raised when progress is reported during solving.
    /// </summary>
    public event EventHandler<ProgressEventArgs>? ProgressChanged;

    /// <summary>
    /// Creates a new Nester2D instance.
    /// </summary>
    public Nester2D()
    {
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Solves a 2D nesting problem.
    /// </summary>
    /// <param name="request">The nesting request.</param>
    /// <returns>The nesting result.</returns>
    /// <exception cref="NestingException">Thrown when solving fails.</exception>
    public NestingResult Solve(NestingRequest request)
    {
        ThrowIfDisposed();

        var requestJson = JsonSerializer.Serialize(request, _jsonOptions);
        int code = NativeLibrary.unesting_solve_2d(requestJson, out var resultPtr);

        return ProcessResult(code, resultPtr);
    }

    /// <summary>
    /// Solves a 2D nesting problem with progress reporting.
    /// </summary>
    /// <param name="request">The nesting request.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The nesting result.</returns>
    /// <exception cref="NestingException">Thrown when solving fails.</exception>
    /// <exception cref="OperationCanceledException">Thrown when cancelled.</exception>
    public NestingResult SolveWithProgress(NestingRequest request, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var requestJson = JsonSerializer.Serialize(request, _jsonOptions);
        var cancelled = false;
        GCHandle? handle = null;

        NativeLibrary.ProgressCallback callback = (progressPtr, userData) =>
        {
            if (cancellationToken.IsCancellationRequested)
            {
                cancelled = true;
                return 0; // Cancel
            }

            try
            {
                var progressJson = Marshal.PtrToStringUTF8(progressPtr);
                if (!string.IsNullOrEmpty(progressJson))
                {
                    var progress = JsonSerializer.Deserialize<ProgressInfo>(progressJson, _jsonOptions);
                    if (progress != null)
                    {
                        var args = new ProgressEventArgs(progress);
                        ProgressChanged?.Invoke(this, args);
                        if (args.Cancel)
                        {
                            cancelled = true;
                            return 0; // Cancel
                        }
                    }
                }
            }
            catch
            {
                // Ignore deserialization errors in callback
            }

            return 1; // Continue
        };

        try
        {
            // Pin the callback delegate
            handle = GCHandle.Alloc(callback);

            int code = NativeLibrary.unesting_solve_2d_with_progress(
                requestJson, callback, IntPtr.Zero, out var resultPtr);

            if (cancelled || code == NativeLibrary.UNESTING_ERR_CANCELLED)
            {
                throw new OperationCanceledException();
            }

            return ProcessResult(code, resultPtr);
        }
        finally
        {
            handle?.Free();
        }
    }

    /// <summary>
    /// Solves a 2D nesting problem asynchronously with progress reporting.
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
            if (progress != null)
            {
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
            }
            else
            {
                return Solve(request);
            }
        }, cancellationToken);
    }

    private NestingResult ProcessResult(int code, IntPtr resultPtr)
    {
        try
        {
            if (code != NativeLibrary.UNESTING_OK)
            {
                throw new NestingException(code, NativeLibrary.GetErrorMessage(code));
            }

            if (resultPtr == IntPtr.Zero)
            {
                throw new NestingException(NativeLibrary.UNESTING_ERR_NULL_PTR, "Null result pointer");
            }

            var resultJson = Marshal.PtrToStringUTF8(resultPtr);
            if (string.IsNullOrEmpty(resultJson))
            {
                throw new NestingException(NativeLibrary.UNESTING_ERR_UNKNOWN, "Empty result");
            }

            var result = JsonSerializer.Deserialize<NestingResult>(resultJson, _jsonOptions);
            return result ?? throw new NestingException(NativeLibrary.UNESTING_ERR_UNKNOWN, "Failed to parse result");
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
            {
                NativeLibrary.unesting_free_string(resultPtr);
            }
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(Nester2D));
        }
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
