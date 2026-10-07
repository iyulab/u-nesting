using System.Text.Json.Serialization;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// Source-generated (de)serialization of requests, results and progress reports — no
/// reflection, so the solvers work in trimmed and NativeAOT hosts.
/// </summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(NestingRequest))]
[JsonSerializable(typeof(NestingResult))]
[JsonSerializable(typeof(PackingRequest))]
[JsonSerializable(typeof(PackingResult))]
[JsonSerializable(typeof(ProgressInfo))]
internal sealed partial class NestingJson : JsonSerializerContext;
