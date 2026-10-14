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

    /// <summary>The catalog as the native library returns it.</summary>
    internal static string StrategiesJson =>
        Marshal.PtrToStringUTF8(NativeLibrary.unesting_strategies())
        ?? throw new InvalidOperationException("The native library returned no strategy catalog.");
}
