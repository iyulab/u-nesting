using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace UNesting;

/// <summary>Reads what a native solve returned, and frees it on every path.</summary>
internal static class NativeCall
{
    /// <summary>
    /// The result on success; <see cref="OperationCanceledException"/> when the caller
    /// cancelled; otherwise the engine's refusal as a <see cref="NestingException"/>
    /// carrying its <see cref="NestingException.Reason"/> and
    /// <see cref="NestingException.Details"/>.
    /// </summary>
    internal static T Read<T>(int code, IntPtr resultPtr, JsonTypeInfo<T> result, bool cancelled = false)
        where T : class
    {
        try
        {
            if (cancelled || code == NativeLibrary.UNESTING_ERR_CANCELLED)
                throw new OperationCanceledException();

            var body = resultPtr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(resultPtr);
            if (code != NativeLibrary.UNESTING_OK)
                throw NestingException.FromResponse(code, body);
            if (string.IsNullOrEmpty(body))
                throw new NestingException(NativeLibrary.UNESTING_ERR_UNKNOWN, "Empty result");

            return JsonSerializer.Deserialize(body, result)
                   ?? throw new NestingException(NativeLibrary.UNESTING_ERR_UNKNOWN, "Failed to parse result");
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
                NativeLibrary.unesting_free_string(resultPtr);
        }
    }
}
