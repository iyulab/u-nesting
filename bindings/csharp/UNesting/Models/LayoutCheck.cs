using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>Whether a 2D layout or 3D packing keeps every rule of its problem, and each rule it breaks.</summary>
/// <param name="Version">The engine version that checked it.</param>
/// <param name="Feasible">No violations.</param>
/// <param name="Violations">Every rule the layout breaks; empty when <paramref name="Feasible"/>.</param>
public sealed record LayoutCheck(string Version, bool Feasible, IReadOnlyList<LayoutViolation> Violations);

/// <summary>
/// One way a layout breaks its problem — the engine's <c>code</c> picks the record.
/// <c>Placement</c> and <c>Other</c> are indices into the layout's placements, <c>Id</c> a
/// geometry id, a rotation in degrees.
/// </summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "code")]
[JsonDerivedType(typeof(UnknownGeometry), "unknown_geometry")]
[JsonDerivedType(typeof(InstanceOutOfRange), "instance_out_of_range")]
[JsonDerivedType(typeof(InstanceRepeated), "instance_repeated")]
[JsonDerivedType(typeof(RotationNotAllowed), "rotation_not_allowed")]
[JsonDerivedType(typeof(FlipNotAllowed), "flip_not_allowed")]
[JsonDerivedType(typeof(OrientationNotAllowed), "orientation_not_allowed")]
[JsonDerivedType(typeof(BinNotAllowed), "bin_not_allowed")]
[JsonDerivedType(typeof(MassExceeded), "mass_exceeded")]
[JsonDerivedType(typeof(Unsupported), "unsupported")]
[JsonDerivedType(typeof(SheetNotAllowed), "sheet_not_allowed")]
[JsonDerivedType(typeof(OutsideBoundary), "outside_boundary")]
[JsonDerivedType(typeof(Overlap), "overlap")]
[JsonDerivedType(typeof(TooClose), "too_close")]
[JsonDerivedType(typeof(MissingFromUnplaced), "missing_from_unplaced")]
[JsonDerivedType(typeof(WronglyUnplaced), "wrongly_unplaced")]
public abstract record LayoutViolation
{
    private LayoutViolation() { }

    /// <summary>The placement names a geometry the problem does not have.</summary>
    public sealed record UnknownGeometry(int Placement, string Id) : LayoutViolation;
    /// <summary>The copy index is not below the geometry's quantity.</summary>
    public sealed record InstanceOutOfRange(int Placement, string Id, int Instance, int Quantity) : LayoutViolation;
    /// <summary>A copy is placed a second time.</summary>
    public sealed record InstanceRepeated(int Placement, string Id, int Instance) : LayoutViolation;
    /// <summary>The rotation (degrees) is not one the geometry allows.</summary>
    public sealed record RotationNotAllowed(int Placement, string Id, double Rotation) : LayoutViolation;
    /// <summary>Flipped, but the geometry does not allow flipping.</summary>
    public sealed record FlipNotAllowed(int Placement, string Id) : LayoutViolation;
    /// <summary>An orientation (its label, <c>"xzy"</c>) the box's constraint does not allow (3D).</summary>
    public sealed record OrientationNotAllowed(int Placement, string Id, string Orientation) : LayoutViolation;
    /// <summary>In a bin after the first: 3D packs one container.</summary>
    public sealed record BinNotAllowed(int Placement, int Bin) : LayoutViolation;
    /// <summary>The packed boxes weigh more than <see cref="Boundary3D.MaxMass"/>.</summary>
    public sealed record MassExceeded(double Mass, double MaxMass) : LayoutViolation;
    /// <summary>A box not supported as gravity or stability requires (3D).</summary>
    public sealed record Unsupported(int Placement, string Id) : LayoutViolation;
    /// <summary>On a sheet after the first without <see cref="Config2D.MultiSheet"/>.</summary>
    public sealed record SheetNotAllowed(int Placement, int Sheet) : LayoutViolation;
    /// <summary>The piece leaves the boundary or comes closer to its edge than the margin.</summary>
    public sealed record OutsideBoundary(int Placement, string Id) : LayoutViolation;
    /// <summary>Two pieces on one sheet overlap.</summary>
    public sealed record Overlap(int Placement, int Other) : LayoutViolation;
    /// <summary>Two pieces on one sheet are closer than the spacing.</summary>
    public sealed record TooClose(int Placement, int Other, double Distance, double Spacing) : LayoutViolation;
    /// <summary>Copies are left unplaced but the geometry is not reported unplaced.</summary>
    public sealed record MissingFromUnplaced(string Id) : LayoutViolation;
    /// <summary>Reported unplaced, but every copy is placed, it is listed twice, or it is unknown.</summary>
    public sealed record WronglyUnplaced(string Id) : LayoutViolation;
}
