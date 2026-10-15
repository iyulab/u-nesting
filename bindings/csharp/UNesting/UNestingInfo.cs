using System.Runtime.InteropServices;
using System.Text.Json;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// Provides information about the U-Nesting library.
/// </summary>
public static class UNestingInfo
{
    private static readonly Lazy<StrategyCatalog> CatalogOfLibrary = new(() =>
        JsonSerializer.Deserialize(StrategiesJson, NestingJson.Default.StrategyCatalog)
        ?? throw new InvalidOperationException("The native library returned no strategy catalog."));

    /// <summary>
    /// Gets the native library version.
    /// </summary>
    public static string Version => NativeLibrary.GetVersion();

    /// <summary>
    /// The strategies each dimension offers and the settings each reads — the table a solve's
    /// refusal of a setting the strategy does not read (<c>invalid_option</c>) comes from.
    /// </summary>
    public static StrategyCatalog Strategies => CatalogOfLibrary.Value;

    /// <summary>
    /// Every refusal <see cref="NestingException.ErrorCode"/> the library and this client
    /// raise (with the fields beside each) and every <see cref="LayoutViolation"/> code the
    /// checkers report — read off the engine's own refusal and violation types, so a caller
    /// that translates codes can check its list against the library's.
    /// </summary>
    public static Codes Codes => CodesOfLibrary.Value;

    private static readonly Lazy<Codes> CodesOfLibrary = new(() =>
        JsonSerializer.Deserialize(CodesJson, NestingJson.Default.Codes)
        ?? throw new InvalidOperationException("The native library returned no code catalog."));

    /// <summary>The code catalog as the native library returns it.</summary>
    internal static string CodesJson =>
        Marshal.PtrToStringUTF8(NativeLibrary.unesting_codes())
        ?? throw new InvalidOperationException("The native library returned no code catalog.");

    /// <summary>The catalog as the native library returns it.</summary>
    internal static string StrategiesJson =>
        Marshal.PtrToStringUTF8(NativeLibrary.unesting_strategies())
        ?? throw new InvalidOperationException("The native library returned no strategy catalog.");
}
