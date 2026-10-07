//! 3D boundary types.

use u_nesting_core::geom::nalgebra_types::NaVector3 as Vector3;
use u_nesting_core::geometry::Boundary;
use u_nesting_core::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 3D boundary (container) for bin packing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Boundary3D {
    /// Dimensions (width, depth, height).
    dimensions: Vector3<f64>,

    /// Maximum total mass allowed.
    max_mass: Option<f64>,

    /// Whether gravity constraints should be enforced.
    gravity: bool,

    /// Whether stability constraints should be enforced.
    stability: bool,
}

impl Boundary3D {
    /// Creates a new 3D boundary with the given dimensions.
    pub fn new(width: f64, depth: f64, height: f64) -> Self {
        Self {
            dimensions: Vector3::new(width, depth, height),
            max_mass: None,
            gravity: false,
            stability: false,
        }
    }

    /// Alias for creating a box-shaped boundary.
    pub fn box_shape(width: f64, depth: f64, height: f64) -> Self {
        Self::new(width, depth, height)
    }

    /// Sets the maximum allowed mass.
    pub fn with_max_mass(mut self, mass: f64) -> Self {
        self.max_mass = Some(mass);
        self
    }

    /// Enables gravity constraints.
    pub fn with_gravity(mut self, enabled: bool) -> Self {
        self.gravity = enabled;
        self
    }

    /// Enables stability constraints.
    pub fn with_stability(mut self, enabled: bool) -> Self {
        self.stability = enabled;
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

    /// Returns the maximum mass.
    pub fn max_mass(&self) -> Option<f64> {
        self.max_mass
    }

    /// Returns whether gravity constraints are enabled.
    pub fn has_gravity(&self) -> bool {
        self.gravity
    }

    /// Returns whether stability constraints are enabled.
    pub fn has_stability(&self) -> bool {
        self.stability
    }
}

impl Boundary for Boundary3D {
    type Scalar = f64;

    fn measure(&self) -> f64 {
        self.dimensions.x * self.dimensions.y * self.dimensions.z
    }

    fn aabb_vec(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 0.0, 0.0],
            vec![self.dimensions.x, self.dimensions.y, self.dimensions.z],
        )
    }

    fn validate(&self) -> Result<()> {
        if self.dimensions.x <= 0.0 || self.dimensions.y <= 0.0 || self.dimensions.z <= 0.0 {
            return Err(Error::invalid_boundary(
                Some("dimensions"),
                "All dimensions must be positive".into(),
            ));
        }

        if let Some(mass) = self.max_mass {
            if mass <= 0.0 {
                return Err(Error::invalid_boundary(
                    Some("max_mass"),
                    "Maximum mass must be positive".into(),
                ));
            }
        }

        Ok(())
    }

    fn contains_point(&self, point: &[f64]) -> bool {
        if point.len() < 3 {
            return false;
        }
        point[0] >= 0.0
            && point[0] <= self.dimensions.x
            && point[1] >= 0.0
            && point[1] <= self.dimensions.y
            && point[2] >= 0.0
            && point[2] <= self.dimensions.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_box_boundary_volume() {
        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        assert_relative_eq!(boundary.measure(), 400000.0, epsilon = 0.001);
    }

    #[test]
    fn test_constraints() {
        let boundary = Boundary3D::new(100.0, 80.0, 50.0)
            .with_max_mass(1000.0)
            .with_gravity(true)
            .with_stability(true);

        assert_eq!(boundary.max_mass(), Some(1000.0));
        assert!(boundary.has_gravity());
        assert!(boundary.has_stability());
    }

    #[test]
    fn test_validation() {
        let valid = Boundary3D::new(100.0, 80.0, 50.0);
        assert!(valid.validate().is_ok());

        let invalid = Boundary3D::new(-100.0, 80.0, 50.0);
        assert!(invalid.validate().is_err());
    }
}
