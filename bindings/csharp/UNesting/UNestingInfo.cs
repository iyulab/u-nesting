namespace UNesting;

/// <summary>
/// Provides information about the U-Nesting library.
/// </summary>
public static class UNestingInfo
{
    /// <summary>
    /// Gets the native library version.
    /// </summary>
    public static string Version => NativeLibrary.GetVersion();
}
