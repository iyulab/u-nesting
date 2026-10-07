using System.Text.Json;

namespace UNesting;

/// <summary>
/// A solve the engine refused or could not finish. <see cref="Exception.Message"/> is
/// readable text; <see cref="Reason"/> and <see cref="Details"/> are for programs.
/// </summary>
public class NestingException : Exception
{
    /// <summary>
    /// The native status: -1 null pointer, -2 malformed input, -3 refused or failed solve,
    /// -99 unknown.
    /// </summary>
    public int ErrorCode { get; }

    /// <summary>
    /// Stable, machine-readable reason, e.g. <c>parameter_out_of_range</c>,
    /// <c>unknown_option</c>, <c>invalid_option</c>, <c>invalid_geometry</c>,
    /// <c>duplicate_id</c>, <c>invalid_boundary</c>, <c>malformed_input</c>,
    /// <c>internal</c>. <c>null</c> when the engine returned no readable body.
    /// </summary>
    public string? Reason { get; }

    /// <summary>
    /// The values behind <see cref="Reason"/>: <c>parameter</c>, <c>id</c>, <c>first</c>,
    /// <c>index</c>, <c>min</c>, <c>max</c>, <c>got</c>, <c>expected</c> — only those the
    /// reason has. <c>null</c> when there are none.
    /// </summary>
    public JsonElement? Details { get; }

    /// <summary>Creates a new NestingException instance.</summary>
    /// <param name="errorCode">The native status.</param>
    /// <param name="message">The error message.</param>
    public NestingException(int errorCode, string message)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    /// <summary>Creates a new NestingException instance with an inner exception.</summary>
    /// <param name="errorCode">The native status.</param>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public NestingException(int errorCode, string message, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = errorCode;
    }

    /// <summary>Creates a refusal with its reason and the values behind it.</summary>
    public NestingException(int errorCode, string message, string? reason, JsonElement? details)
        : base(message)
    {
        ErrorCode = errorCode;
        Reason = reason;
        Details = details;
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
            string? reason = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            JsonElement? details = root.TryGetProperty("details", out var d) && d.ValueKind == JsonValueKind.Object
                ? d.Clone()
                : null;
            return new NestingException(status, message, reason, details);
        }
        catch (JsonException)
        {
            return new NestingException(status, body);
        }
    }
}
