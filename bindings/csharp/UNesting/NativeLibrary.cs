using System;
using System.Runtime.InteropServices;

namespace UNesting;

/// <summary>
/// P/Invoke declarations for the U-Nesting native library.
/// </summary>
internal static class NativeLibrary
{
    private const string LibraryName = "u_nesting_ffi";

    #region Error Codes

    public const int UNESTING_OK = 0;
    public const int UNESTING_ERR_NULL_PTR = -1;
    public const int UNESTING_ERR_INVALID_JSON = -2;
    public const int UNESTING_ERR_SOLVE_FAILED = -3;
    public const int UNESTING_ERR_CANCELLED = -4;
    public const int UNESTING_ERR_UNKNOWN = -99;

    #endregion

    #region Callback Delegate

    /// <summary>
    /// Progress callback delegate.
    /// </summary>
    /// <param name="progressJson">JSON string containing progress information.</param>
    /// <param name="userData">User-provided context pointer.</param>
    /// <returns>Non-zero to continue, zero to cancel.</returns>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate int ProgressCallback(IntPtr progressJson, IntPtr userData);

    #endregion

    #region Basic API

    /// <summary>
    /// Solves a 2D nesting problem.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int unesting_solve_2d(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        out IntPtr resultPtr);

    /// <summary>
    /// Solves a 3D bin packing problem.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int unesting_solve_3d(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        out IntPtr resultPtr);

    /// <summary>
    /// Auto-detects mode and solves.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int unesting_solve(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        out IntPtr resultPtr);

    #endregion

    #region Progress Callback API

    /// <summary>
    /// Solves a 2D nesting problem with progress callback.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int unesting_solve_2d_with_progress(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        ProgressCallback? callback,
        IntPtr userData,
        out IntPtr resultPtr);

    /// <summary>
    /// Solves a 3D bin packing problem with progress callback.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int unesting_solve_3d_with_progress(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        ProgressCallback? callback,
        IntPtr userData,
        out IntPtr resultPtr);

    /// <summary>
    /// Auto-detects mode and solves with progress callback.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int unesting_solve_with_progress(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
        ProgressCallback? callback,
        IntPtr userData,
        out IntPtr resultPtr);

    #endregion

    #region Utility

    /// <summary>
    /// Frees a string allocated by the native library.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void unesting_free_string(IntPtr ptr);

    /// <summary>
    /// Returns the API version string.
    /// </summary>
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr unesting_version();

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the library version.
    /// </summary>
    public static string GetVersion()
    {
        var ptr = unesting_version();
        return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
    }

    /// <summary>
    /// Converts an error code to a descriptive message.
    /// </summary>
    public static string GetErrorMessage(int errorCode) => errorCode switch
    {
        UNESTING_OK => "Success",
        UNESTING_ERR_NULL_PTR => "Null pointer passed",
        UNESTING_ERR_INVALID_JSON => "Invalid JSON input",
        UNESTING_ERR_SOLVE_FAILED => "Solver failed",
        UNESTING_ERR_CANCELLED => "Operation cancelled",
        UNESTING_ERR_UNKNOWN => "Unknown error",
        _ => $"Unknown error code: {errorCode}"
    };

    #endregion
}
