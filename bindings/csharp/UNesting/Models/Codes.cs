namespace UNesting.Models;

/// <summary>A refusal reason: the <c>code</c> an exception's <c>ErrorCode</c> carries and the
/// <c>details</c> fields that can stand beside it.</summary>
/// <param name="Code">The stable reason, e.g. <c>parameter_out_of_range</c>.</param>
/// <param name="Fields">The fields beside it (<c>parameter</c>, <c>min</c>, …).</param>
public sealed record RefusalCode(string Code, IReadOnlyList<string> Fields);

/// <summary>A violation a checker reports: its <c>code</c>, the fields beside it, and the
/// engine operations that report it.</summary>
/// <param name="Code">The stable reason, e.g. <c>overlap</c>.</param>
/// <param name="Fields">The violation's fields beside <c>code</c>, in wire names.</param>
/// <param name="Checkers">The engine operations reporting it, by their engine names
/// (<c>check_2d</c> is <c>Nester2D.Check</c>, <c>check_3d</c> is <c>Packer3D.Check</c>).</param>
public sealed record ViolationCode(string Code, IReadOnlyList<string> Fields, IReadOnlyList<string> Checkers);

/// <summary>Every code the library can answer with besides a result — the same catalog the
/// WebAssembly package's <c>codes()</c> returns.</summary>
/// <param name="Refusals">Each refusal reason.</param>
/// <param name="Violations">Each checker violation.</param>
public sealed record Codes(IReadOnlyList<RefusalCode> Refusals, IReadOnlyList<ViolationCode> Violations);
