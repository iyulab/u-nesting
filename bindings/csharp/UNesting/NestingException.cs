using System.Text.Json;

namespace UNesting;

/// <summary>
/// A solve the engine refused or could not finish. <see cref="Exception.Message"/> is readable
/// text; <see cref="ErrorCode"/>,
/// <see cref="Parameter"/>, <see cref="Index"/> and <see cref="Details"/> are for programs —
/// the same members, with the same meaning, on every sibling package's exception.
/// </summary>
public class NestingException : Exception
{
    /// <summary>
    /// The native status: -1 null pointer, -2 malformed input, -3 refused or failed solve,
    /// -99 unknown.
    /// </summary>
    public int Status { get; }

    /// <summary>
    /// Stable, machine-readable reason: the <c>code</c> of the engine's error body, the same
    /// name the WebAssembly binding puts on its <c>Error</c> —
    /// <c>parameter_out_of_range</c>, <c>empty_input</c>, <c>unknown_option</c>,
    /// <c>invalid_option</c>, <c>invalid_geometry</c>, <c>duplicate_id</c>,
    /// <c>invalid_boundary</c>, <c>malformed_input</c>, <c>internal</c> — and
    /// <c>value_not_finite</c>, which the client raises itself for a NaN or infinite number
    /// before the request is sent.
    /// <c>null</c> when the engine returned no readable body.
    /// </summary>
    public string? ErrorCode { get; }

    /// <summary>
    /// The argument or option the refusal is about, as a path into the request
    /// (<c>config.time_limit_ms</c>, <c>geometries[0].polygon</c>, …) — the body's
    /// <c>parameter</c>. <c>null</c> when the refusal is not
    /// about one value.
    /// </summary>
    public string? Parameter { get; }

    /// <summary>
    /// Zero-based position of the offending element in the array <see cref="Parameter"/>
    /// names — the body's <c>index</c>. <c>null</c> when the value is not an array element.
    /// </summary>
    public int? Index { get; }

    /// <summary>
    /// The values behind <see cref="ErrorCode"/> — the body's <c>details</c>: <c>parameter</c>,
    /// <c>id</c>, <c>first</c>, <c>index</c>, <c>min</c>, <c>max</c>, <c>exclusive_min</c>,
    /// <c>exclusive_max</c>, <c>got</c>, <c>expected</c>, only those the reason has. <c>min</c> /
    /// <c>max</c> are closed bounds (the value may equal them), <c>exclusive_min</c> /
    /// <c>exclusive_max</c> open ones; a side has at most one of the two. <c>null</c> when there
    /// are none.
    /// </summary>
    public JsonElement? Details { get; }

    /// <summary>A failure with no readable body.</summary>
    /// <param name="status">The native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    public NestingException(int status, string message)
        : base(message)
    {
        Status = status;
    }

    /// <summary>A failure with no readable body, caused by <paramref
    /// name="innerException"/>.</summary>
    /// <param name="status">The native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    /// <param name="innerException">The cause.</param>
    public NestingException(int status, string message, Exception innerException)
        : base(message, innerException)
    {
        Status = status;
    }

    /// <summary>
    /// A refusal with its reason and the values behind it; <see cref="Parameter"/> and
    /// <see cref="Index"/> are read from <paramref name="details"/>.
    /// </summary>
    /// <param name="status">The native status, see <see cref="Status"/>.</param>
    /// <param name="message">What went wrong, in words.</param>
    /// <param name="errorCode">The body's <c>code</c>.</param>
    /// <param name="details">The body's <c>details</c>.</param>
    public NestingException(int status, string message, string? errorCode, JsonElement? details)
        : base(message)
    {
        Status = status;
        ErrorCode = errorCode;
        Details = details;
        Parameter = ParameterIn(details);
        Index = IndexIn(details);
    }

    /// <summary>
    /// Reads the engine's refusal: the response's <c>error</c> text, its <c>code</c> and
    /// its <c>details</c>. Without a body, the status alone is all there is to say.
    /// </summary>
    internal static NestingException FromResponse(int status, string? body)
    {
        if (string.IsNullOrEmpty(body))
            return new NestingException(status, NativeLibrary.GetErrorMessage(status));
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            var message = root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : NativeLibrary.GetErrorMessage(status);
            string? errorCode = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            JsonElement? details = root.TryGetProperty("details", out var d) && d.ValueKind == JsonValueKind.Object
                ? d.Clone()
                : null;
            return new NestingException(status, message, errorCode, details);
        }
        catch (JsonException)
        {
            return new NestingException(status, body);
        }
    }

    /// <summary>The body's <c>parameter</c>, when it is a string.</summary>
    private static string? ParameterIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("parameter", out var p) && p.ValueKind == JsonValueKind.String
            ? p.GetString()
            : null;

    /// <summary>The body's <c>index</c>, when it is a whole number.</summary>
    private static int? IndexIn(JsonElement? details) =>
        details is { ValueKind: JsonValueKind.Object } d
        && d.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number
        && i.TryGetInt32(out var n)
            ? n
            : null;
}
