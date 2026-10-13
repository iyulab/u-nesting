using System.Text.Json.Serialization;
using UNesting.Models;

namespace UNesting;

/// <summary>
/// Source-generated (de)serialization of requests, results and progress reports — no
/// reflection, so the solvers work in trimmed and NativeAOT hosts. A key the engine does
/// not send, or a null where a record does not allow one, fails the call instead of
/// reading as a default. Request properties that are <c>null</c> are left out, so the
/// engine's default applies.
/// </summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(NestingRequest))]
[JsonSerializable(typeof(NestingResult))]
[JsonSerializable(typeof(PackingRequest))]
[JsonSerializable(typeof(PackingResult))]
[JsonSerializable(typeof(ProgressInfo))]
internal sealed partial class NestingJson : JsonSerializerContext;
