using System.Runtime.InteropServices;
using System.Text.Json;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// 3D bin packing solver for box placement optimization.
/// </summary>
public class Packer3D : IDisposable
{
    private readonly JsonSerializerOptions _jsonOptions;
    private bool _disposed;

    /// <summary>
    /// Event raised when progress is reported during solving.
    /// </summary>
    public event EventHandler<ProgressEventArgs>? ProgressChanged;

    /// <summary>
    /// Creates a new Packer3D instance.
    /// </summary>
    public Packer3D()
    {
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Solves a 3D bin packing problem.
    /// </summary>
    /// <param name="request">The packing request.</param>
    /// <returns>The packing result.</returns>
    /// <exception cref="NestingException">Thrown when solving fails.</exception>
    public PackingResult Solve(PackingRequest request)
    {
        ThrowIfDisposed();

        var requestJson = JsonSerializer.Serialize(request, _jsonOptions);
        int code = NativeLibrary.unesting_solve_3d(requestJson, out var resultPtr);

        return ProcessResult(code, resultPtr);
    }

    /// <summary>
    /// Solves a 3D bin packing problem with progress reporting.
    /// </summary>
    /// <param name="request">The packing request.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The packing result.</returns>
    /// <exception cref="NestingException">Thrown when solving fails.</exception>
    /// <exception cref="OperationCanceledException">Thrown when cancelled.</exception>
    public PackingResult SolveWithProgress(PackingRequest request, CancellationToken cancellationToken = default)
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

            int code = NativeLibrary.unesting_solve_3d_with_progress(
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
    /// Solves a 3D bin packing problem asynchronously with progress reporting.
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

    private PackingResult ProcessResult(int code, IntPtr resultPtr)
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

            var result = JsonSerializer.Deserialize<PackingResult>(resultJson, _jsonOptions);
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
            throw new ObjectDisposedException(nameof(Packer3D));
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
