using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using UNesting.Models;

namespace UNesting;

/// <summary>One native solve: the request out, the result (and progress reports) back.</summary>
internal static class NativeCall
{
    internal delegate int Solve(string requestJson, out IntPtr resultPtr);

    internal delegate int SolveWithProgress(
        string requestJson, NativeLibrary.ProgressCallback? callback, IntPtr userData, out IntPtr resultPtr);

    /// <summary>Raw response and progress bodies, as the engine wrote them — the contract tests compare them with the records.</summary>
    internal sealed class Observers
    {
        internal Action<string>? Response { get; set; }
        internal Action<string>? Progress { get; set; }
    }

    internal static T Run<T>(Solve solve, string requestJson, JsonTypeInfo<T> result, Observers observers)
        where T : class
    {
        var code = solve(requestJson, out var resultPtr);
        return Read(code, resultPtr, result, observers, cancelled: false, failure: null);
    }

    /// <summary>
    /// Solves while reporting progress to <paramref name="report"/>, which returns true to
    /// cancel. An exception thrown while reading or handling a report stops the solve and is
    /// rethrown here, once the engine has returned — it cannot cross the native frame.
    /// </summary>
    internal static T Run<T>(
        SolveWithProgress solve,
        string requestJson,
        JsonTypeInfo<T> result,
        Observers observers,
        Func<ProgressInfo, bool> report,
        CancellationToken cancellationToken)
        where T : class
    {
        var cancelled = false;
        ExceptionDispatchInfo? failure = null;

        NativeLibrary.ProgressCallback callback = (progressPtr, _) =>
        {
            if (cancellationToken.IsCancellationRequested)
            {
                cancelled = true;
                return 0;
            }
            try
            {
                var body = Marshal.PtrToStringUTF8(progressPtr);
                if (string.IsNullOrEmpty(body))
                    return 1;
                observers.Progress?.Invoke(body);
                var progress = JsonSerializer.Deserialize(body, NestingJson.Default.ProgressInfo)
                               ?? throw new JsonException("The engine sent a null progress report.");
                if (report(progress))
                {
                    cancelled = true;
                    return 0;
                }
                return 1;
            }
            catch (Exception e)
            {
                failure = ExceptionDispatchInfo.Capture(e);
                return 0;
            }
        };

        var code = solve(requestJson, callback, IntPtr.Zero, out var resultPtr);
        GC.KeepAlive(callback);
        return Read(code, resultPtr, result, observers, cancelled, failure);
    }

    /// <summary>
    /// The result on success; otherwise the failure a progress handler raised,
    /// <see cref="OperationCanceledException"/> when the caller cancelled, or the engine's
    /// refusal as a <see cref="NestingException"/> carrying its
    /// <see cref="NestingException.Reason"/> and <see cref="NestingException.Details"/>.
    /// The native string is freed on every path.
    /// </summary>
    private static T Read<T>(
        int code, IntPtr resultPtr, JsonTypeInfo<T> result, Observers observers,
        bool cancelled, ExceptionDispatchInfo? failure)
        where T : class
    {
        try
        {
            failure?.Throw();
            if (cancelled || code == NativeLibrary.UNESTING_ERR_CANCELLED)
                throw new OperationCanceledException();

            var body = resultPtr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(resultPtr);
            if (code != NativeLibrary.UNESTING_OK)
                throw NestingException.FromResponse(code, body);
            if (string.IsNullOrEmpty(body))
                throw new NestingException(NativeLibrary.UNESTING_ERR_UNKNOWN, "Empty result");

            observers.Response?.Invoke(body);
            return JsonSerializer.Deserialize(body, result)
                   ?? throw new NestingException(NativeLibrary.UNESTING_ERR_UNKNOWN, "The engine returned null.");
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
                NativeLibrary.unesting_free_string(resultPtr);
        }
    }
}
