//! Core geometry traits and types.

use crate::transform::{AABB2D, AABB3D};
use crate::Result;
use nalgebra::RealField;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for a geometry.
pub type GeometryId = String;

/// Allowed rotation angles for a geometry.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum RotationConstraint<S> {
    /// No rotation allowed (fixed orientation).
    #[default]
    None,
    /// Free rotation (any angle).
    Free,
    /// Discrete rotation steps (e.g., 0, 90, 180, 270 degrees).
    Discrete(Vec<S>),
}

impl<S: RealField + Copy> RotationConstraint<S> {
    /// Creates a constraint for axis-aligned rotations only (0, 90, 180, 270 degrees).
    pub fn axis_aligned() -> Self {
        let pi = S::pi();
        let half_pi = pi / (S::one() + S::one());
        Self::Discrete(vec![S::zero(), half_pi, pi, pi + half_pi])
    }

    /// Creates a constraint for n evenly-spaced rotations.
    ///
    /// # Panics
    /// Panics if `n` exceeds the precision of the scalar type (unlikely in practice:
    /// n > 2^24 for f32, n > 2^53 for f64).
    pub fn steps(n: usize) -> Self {
        if n == 0 {
            return Self::None;
        }
        let two_pi = S::two_pi();
        let step =
            two_pi / S::from_usize(n).expect("n exceeds scalar precision (use n < 2^24 for f32)");
        let angles: Vec<S> = (0..n)
            .map(|i| step * S::from_usize(i).expect("index exceeds scalar precision"))
            .collect();
        Self::Discrete(angles)
    }

    /// Returns true if no rotation is allowed.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns the list of allowed angles.
    pub fn angles(&self) -> Vec<S> {
        match self {
            Self::None => vec![S::zero()],
            Self::Free => vec![], // Empty means any angle
            Self::Discrete(angles) => angles.clone(),
        }
    }
}

/// Trait for geometric shapes that can be nested or packed.
pub trait Geometry: Clone + Send + Sync {
    /// The coordinate type (f32 or f64).
    type Scalar: RealField + Copy;

    /// Returns the unique identifier for this geometry.
    fn id(&self) -> &GeometryId;

    /// Returns the quantity of this geometry to place.
    fn quantity(&self) -> usize;

    /// Returns the area (2D) or volume (3D) of this geometry.
    fn measure(&self) -> Self::Scalar;

    /// Returns the axis-aligned bounding box as (min, max) corners.
    fn aabb(&self) -> ([Self::Scalar; 2], [Self::Scalar; 2]) {
        // Default implementation for 2D, override for 3D
        let (min, max) = self.aabb_vec();
        ([min[0], min[1]], [max[0], max[1]])
    }

    /// Returns the axis-aligned bounding box as Vec (for generic dimension support).
    fn aabb_vec(&self) -> (Vec<Self::Scalar>, Vec<Self::Scalar>);

    /// Returns the centroid (center of mass) of this geometry.
    fn centroid(&self) -> Vec<Self::Scalar>;

    /// Validates the geometry and returns an error if invalid.
    fn validate(&self) -> Result<()>;

    /// Returns the allowed rotations for this geometry.
    fn rotation_constraint(&self) -> &RotationConstraint<Self::Scalar>;

    /// Returns whether mirroring/flipping is allowed.
    fn allow_mirror(&self) -> bool {
        false
    }

    /// Returns optional priority for placement order (higher = placed first).
    fn priority(&self) -> i32 {
        0
    }
}

/// Extended trait for 2D geometries.
pub trait Geometry2DExt: Geometry {
    /// Returns the 2D AABB.
    fn aabb_2d(&self) -> AABB2D<Self::Scalar>;

    /// Returns the outer boundary as a sequence of points (polygon vertices).
    fn outer_ring(&self) -> &[(Self::Scalar, Self::Scalar)];

    /// Returns any holes in the geometry.
    fn holes(&self) -> &[Vec<(Self::Scalar, Self::Scalar)>];

    /// Returns true if this geometry has holes.
    fn has_holes(&self) -> bool {
        !self.holes().is_empty()
    }

    /// Returns true if this geometry is convex.
    fn is_convex(&self) -> bool;

    /// Returns the convex hull of this geometry.
    fn convex_hull(&self) -> Vec<(Self::Scalar, Self::Scalar)>;

    /// Returns the perimeter of this geometry.
    fn perimeter(&self) -> Self::Scalar;
}

/// Extended trait for 3D geometries.
pub trait Geometry3DExt: Geometry {
    /// Returns the 3D AABB.
    fn aabb_3d(&self) -> AABB3D<Self::Scalar>;

    /// Returns the surface area of this geometry.
    fn surface_area(&self) -> Self::Scalar;

    /// Returns the mass of this geometry, if defined.
    fn mass(&self) -> Option<Self::Scalar>;

    /// Returns the center of mass of this geometry.
    fn center_of_mass(&self) -> (Self::Scalar, Self::Scalar, Self::Scalar);

    /// Returns whether this geometry can be stacked upon.
    fn stackable(&self) -> bool {
        true
    }

    /// Returns the maximum stacking load this geometry can support.
    fn max_stack_load(&self) -> Option<Self::Scalar> {
        None
    }
}

/// Trait for boundaries/containers that hold geometries.
pub trait Boundary: Clone + Send + Sync {
    /// The coordinate type (f32 or f64).
    type Scalar: RealField + Copy;

    /// Returns the area (2D) or volume (3D) of this boundary.
    fn measure(&self) -> Self::Scalar;

    /// Returns the axis-aligned bounding box as (min, max) corners.
    fn aabb(&self) -> ([Self::Scalar; 2], [Self::Scalar; 2]) {
        let (min, max) = self.aabb_vec();
        ([min[0], min[1]], [max[0], max[1]])
    }

    /// Returns the axis-aligned bounding box as Vec.
    fn aabb_vec(&self) -> (Vec<Self::Scalar>, Vec<Self::Scalar>);

    /// Validates the boundary and returns an error if invalid.
    fn validate(&self) -> Result<()>;

    /// Checks if a point is inside the boundary.
    fn contains_point(&self, point: &[Self::Scalar]) -> bool;
}

/// Extended trait for 2D boundaries.
pub trait Boundary2DExt: Boundary {
    /// Returns the 2D AABB.
    fn aabb_2d(&self) -> AABB2D<Self::Scalar>;

    /// Returns the boundary polygon vertices.
    fn vertices(&self) -> &[(Self::Scalar, Self::Scalar)];

    /// Checks if a polygon is fully contained within this boundary.
    fn contains_polygon(&self, polygon: &[(Self::Scalar, Self::Scalar)]) -> bool;

    /// Returns the effective usable area after applying margin.
    fn effective_area(&self, margin: Self::Scalar) -> Self::Scalar;
}

/// Extended trait for 3D boundaries.
pub trait Boundary3DExt: Boundary {
    /// Returns the 3D AABB.
    fn aabb_3d(&self) -> AABB3D<Self::Scalar>;

    /// Returns the maximum weight/mass capacity.
    fn max_mass(&self) -> Option<Self::Scalar>;

    /// Checks if a box is fully contained within this boundary.
    fn contains_box(&self, min: &[Self::Scalar; 3], max: &[Self::Scalar; 3]) -> bool;

    /// Returns the effective usable volume after applying margin.
    fn effective_volume(&self, margin: Self::Scalar) -> Self::Scalar;
}

/// Orientation constraints for 3D packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum Orientation3D {
    /// Original orientation only (no rotation).
    Fixed,
    /// Any of the 6 axis-aligned orientations.
    #[default]
    AxisAligned,
    /// Any of the 24 orthogonal orientations.
    Orthogonal,
    /// Free rotation (any orientation).
    Free,
}

impl Orientation3D {
    /// Returns the number of discrete orientations.
    pub fn count(&self) -> usize {
        match self {
            Self::Fixed => 1,
            Self::AxisAligned => 6,
            Self::Orthogonal => 24,
            Self::Free => usize::MAX, // Continuous
        }
    }

    /// Returns true if rotation is completely fixed.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_constraint_axis_aligned() {
        let constraint: RotationConstraint<f64> = RotationConstraint::axis_aligned();
        let angles = constraint.angles();
        assert_eq!(angles.len(), 4);
    }

    #[test]
    fn test_rotation_constraint_steps() {
        let constraint: RotationConstraint<f64> = RotationConstraint::steps(8);
        let angles = constraint.angles();
        assert_eq!(angles.len(), 8);
    }

    #[test]
    fn test_orientation_3d_count() {
        assert_eq!(Orientation3D::Fixed.count(), 1);
        assert_eq!(Orientation3D::AxisAligned.count(), 6);
        assert_eq!(Orientation3D::Orthogonal.count(), 24);
    }
}
