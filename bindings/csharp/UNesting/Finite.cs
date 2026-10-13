using System.Globalization;
using System.Text.Json.Nodes;

namespace UNesting;

/// <summary>
/// JSON has no NaN or infinity, so such a number cannot reach the engine. It is refused
/// before the request is written, with the engine's reason (<c>value_not_finite</c>) and
/// fields: <c>parameter</c> the path in the request to the value — or to the innermost
/// array holding it, with <c>index</c> its position there — and <c>id</c> the geometry it
/// belongs to.
/// </summary>
internal static class Finite
{
    internal static void Value(string parameter, double? value, string? id = null)
    {
        if (value is { } v && !double.IsFinite(v))
            throw Refusal(parameter, v, null, id);
    }

    internal static void Values(string parameter, IReadOnlyList<double> values, string? id = null)
    {
        for (var i = 0; i < values.Count; i++)
        {
            if (!double.IsFinite(values[i]))
                throw Refusal(parameter, values[i], i, id);
        }
    }

    /// <summary>Vertices <c>[[x, y], ...]</c>: a bad coordinate is at <c>parameter[vertex]</c>, <c>index</c> 0 for x, 1 for y.</summary>
    internal static void Points(string parameter, IReadOnlyList<double[]> points, string? id = null)
    {
        for (var v = 0; v < points.Count; v++)
        {
            if (points[v] is { } point)
                Values($"{parameter}[{v}]", point, id);
        }
    }

    private static NestingException Refusal(string parameter, double value, int? index, string? id)
    {
        var where = index is { } i ? $"{parameter}[{i}]" : parameter;
        var details = new JsonObject { ["parameter"] = parameter };
        if (id is not null)
            details["id"] = id;
        if (index is not null)
            details["index"] = index;
        var body = new JsonObject
        {
            ["error"] = $"{where}: expected a finite number, got {value.ToString(CultureInfo.InvariantCulture)}",
            ["code"] = "value_not_finite",
            ["details"] = details,
        };
        return NestingException.FromResponse(NativeLibrary.UNESTING_ERR_SOLVE_FAILED, body.ToJsonString());
    }
}
