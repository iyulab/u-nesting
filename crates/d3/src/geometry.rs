//! 3D geometry types.

use u_nesting_core::geom::nalgebra_types::NaVector3 as Vector3;
use u_nesting_core::geometry::{Geometry, GeometryId, RotationConstraint};
use u_nesting_core::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Orientation constraint for 3D placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OrientationConstraint {
    /// Any orientation allowed (6 axis-aligned rotations for boxes).
    #[default]
    Any,
    /// Only upright orientations (2 rotations: original and 90° around Z).
    Upright,
    /// Fixed orientation (no rotation allowed).
    Fixed,
}

/// A 3D box geometry that can be packed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Geometry3D {
    /// Unique identifier.
    id: GeometryId,

    /// Dimensions (width, depth, height).
    dimensions: Vector3<f64>,

    /// Number of copies to place.
    quantity: usize,

    /// Mass of the geometry (for weight constraints).
    mass: Option<f64>,

    /// Orientation constraint (for 3D-specific orientations).
    orientation: OrientationConstraint,

    /// Rotation constraint (for Geometry trait).
    rotation_constraint: RotationConstraint<f64>,

    /// Whether this item can have other items stacked on top.
    stackable: bool,

    /// Maximum weight that can be placed on top of this item.
    max_stack_weight: Option<f64>,
}

impl Geometry3D {
    /// Creates a new 3D box geometry with the given ID and dimensions.
    pub fn new(id: impl Into<GeometryId>, width: f64, depth: f64, height: f64) -> Self {
        Self {
            id: id.into(),
            dimensions: Vector3::new(width, depth, height),
            quantity: 1,
            mass: None,
            orientation: OrientationConstraint::default(),
            rotation_constraint: RotationConstraint::None,
            stackable: true,
            max_stack_weight: None,
        }
    }

    /// Alias for creating a box shape.
    pub fn box_shape(id: impl Into<GeometryId>, width: f64, depth: f64, height: f64) -> Self {
        Self::new(id, width, depth, height)
    }

    /// Sets the quantity to place.
    pub fn with_quantity(mut self, n: usize) -> Self {
        self.quantity = n;
        self
    }

    /// Sets the mass.
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = Some(mass);
        self
    }

    /// Sets the orientation constraint.
    pub fn with_orientation(mut self, constraint: OrientationConstraint) -> Self {
        self.orientation = constraint;
        self
    }

    /// Sets whether items can be stacked on top.
    pub fn with_stackable(mut self, stackable: bool) -> Self {
        self.stackable = stackable;
        self
    }

    /// Sets the maximum weight that can be stacked on top.
    pub fn with_max_stack_weight(mut self, weight: f64) -> Self {
        self.max_stack_weight = Some(weight);
        self
    }

    /// Returns the dimensions (width, depth, height).
    pub fn dimensions(&self) -> &Vector3<f64> {
        &self.dimensions
    }

    /// Returns the width.
    pub fn width(&self) -> f64 {
        self.dimensions.x
    }

    /// Returns the depth.
    pub fn depth(&self) -> f64 {
        self.dimensions.y
    }

    /// Returns the height.
    pub fn height(&self) -> f64 {
        self.dimensions.z
    }

    /// Returns the mass.
    pub fn mass(&self) -> Option<f64> {
        self.mass
    }

    /// Returns the orientation constraint.
    pub fn orientation_constraint(&self) -> OrientationConstraint {
        self.orientation
    }

    /// Returns whether items can be stacked on top.
    pub fn is_stackable(&self) -> bool {
        self.stackable
    }

    /// Returns the allowed orientations based on the constraint.
    /// Each orientation is (width_axis, depth_axis, height_axis).
    pub fn allowed_orientations(&self) -> Vec<(usize, usize, usize)> {
        match self.orientation {
            OrientationConstraint::Fixed => vec![(0, 1, 2)],
            OrientationConstraint::Upright => vec![(0, 1, 2), (1, 0, 2)],
            OrientationConstraint::Any => vec![
                (0, 1, 2), // Original
                (0, 2, 1), // Rotated 90° around X
                (1, 0, 2), // Rotated 90° around Z
                (1, 2, 0), // Rotated 90° around X then Z
                (2, 0, 1), // Rotated 90° around Y
                (2, 1, 0), // Rotated 90° around Y then X
            ],
        }
    }

    /// Returns the orientation label for a given orientation index, as an
    /// axis-permutation string (e.g. `"xyz"`, `"xzy"`).
    ///
    /// The permutation `(x_idx, y_idx, z_idx)` from [`allowed_orientations`]
    /// is mapped to the axis letters `x`/`y`/`z`. Out-of-range indices fall
    /// back to the identity orientation `"xyz"`.
    ///
    /// [`allowed_orientations`]: Self::allowed_orientations
    pub fn orientation_label(&self, orientation: usize) -> String {
        const AXES: [char; 3] = ['x', 'y', 'z'];
        let orientations = self.allowed_orientations();
        let (a, b, c) = orientations.get(orientation).copied().unwrap_or((0, 1, 2));
        [AXES[a], AXES[b], AXES[c]].iter().collect()
    }

    /// Returns dimensions for a given orientation index.
    pub fn dimensions_for_orientation(&self, orientation: usize) -> Vector3<f64> {
        let orientations = self.allowed_orientations();
        if orientation >= orientations.len() {
            return self.dimensions;
        }

        let (x_idx, y_idx, z_idx) = orientations[orientation];
        Vector3::new(
            self.dimensions[x_idx],
            self.dimensions[y_idx],
            self.dimensions[z_idx],
        )
    }
}

impl Geometry for Geometry3D {
    type Scalar = f64;

    fn id(&self) -> &GeometryId {
        &self.id
    }

    fn quantity(&self) -> usize {
        self.quantity
    }

    fn measure(&self) -> f64 {
        self.dimensions.x * self.dimensions.y * self.dimensions.z
    }

    fn aabb_vec(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 0.0, 0.0],
            vec![self.dimensions.x, self.dimensions.y, self.dimensions.z],
        )
    }

    fn centroid(&self) -> Vec<f64> {
        vec![
            self.dimensions.x / 2.0,
            self.dimensions.y / 2.0,
            self.dimensions.z / 2.0,
        ]
    }

    fn rotation_constraint(&self) -> &RotationConstraint<f64> {
        &self.rotation_constraint
    }

    fn validate(&self) -> Result<()> {
        if self.dimensions.x <= 0.0 || self.dimensions.y <= 0.0 || self.dimensions.z <= 0.0 {
            return Err(Error::InvalidGeometry(format!(
                "All dimensions for '{}' must be positive",
                self.id
            )));
        }

        if self.quantity == 0 {
            return Err(Error::InvalidGeometry(format!(
                "Quantity for '{}' must be at least 1",
                self.id
            )));
        }

        if let Some(mass) = self.mass {
            if mass < 0.0 {
                return Err(Error::InvalidGeometry(format!(
                    "Mass for '{}' cannot be negative",
                    self.id
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_box_volume() {
        let box3d = Geometry3D::new("B1", 10.0, 20.0, 30.0);
        assert_relative_eq!(box3d.measure(), 6000.0, epsilon = 0.001);
    }

    #[test]
    fn test_orientations() {
        let box3d = Geometry3D::new("B1", 10.0, 20.0, 30.0);

        // Any orientation should have 6 options
        assert_eq!(box3d.allowed_orientations().len(), 6);

        // Upright should have 2 options
        let upright = box3d
            .clone()
            .with_orientation(OrientationConstraint::Upright);
        assert_eq!(upright.allowed_orientations().len(), 2);

        // Fixed should have 1 option
        let fixed = box3d.clone().with_orientation(OrientationConstraint::Fixed);
        assert_eq!(fixed.allowed_orientations().len(), 1);
    }

    #[test]
    fn test_orientation_label() {
        // Default constraint is Any → 6 axis permutations map to labels.
        let geom = Geometry3D::new("B1", 10.0, 20.0, 30.0);
        assert_eq!(geom.orientation_label(0), "xyz");
        assert_eq!(geom.orientation_label(1), "xzy");
        assert_eq!(geom.orientation_label(2), "yxz");
        assert_eq!(geom.orientation_label(3), "yzx");
        assert_eq!(geom.orientation_label(4), "zxy");
        assert_eq!(geom.orientation_label(5), "zyx");
        // Out-of-range index falls back to identity orientation.
        assert_eq!(geom.orientation_label(99), "xyz");

        // Fixed constraint only ever yields the identity orientation.
        let fixed = geom.clone().with_orientation(OrientationConstraint::Fixed);
        assert_eq!(fixed.orientation_label(0), "xyz");
    }

    #[test]
    fn test_aabb() {
        use u_nesting_core::geometry::Geometry;
        let box3d = Geometry3D::new("B1", 10.0, 20.0, 30.0);
        let (min, max) = box3d.aabb_vec();

        assert_eq!(min, vec![0.0, 0.0, 0.0]);
        assert_eq!(max, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_validation() {
        let valid = Geometry3D::new("B1", 10.0, 20.0, 30.0);
        assert!(valid.validate().is_ok());

        let invalid = Geometry3D::new("B2", -10.0, 20.0, 30.0);
        assert!(invalid.validate().is_err());

        let zero_qty = Geometry3D::new("B3", 10.0, 20.0, 30.0).with_quantity(0);
        assert!(zero_qty.validate().is_err());
    }
}
