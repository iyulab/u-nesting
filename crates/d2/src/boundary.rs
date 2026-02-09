//! 2D boundary types.

use u_nesting_core::geom::polygon as geom_polygon;
use u_nesting_core::geometry::{Boundary, Boundary2DExt};
use u_nesting_core::transform::AABB2D;
use u_nesting_core::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 2D boundary (container) for nesting.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Boundary2D {
    /// Boundary shape as polygon vertices.
    exterior: Vec<(f64, f64)>,

    /// Interior obstacles/holes (regions where placement is forbidden).
    holes: Vec<Vec<(f64, f64)>>,

    /// Width (for rectangular boundaries).
    width: Option<f64>,

    /// Height (for rectangular boundaries).
    height: Option<f64>,

    /// Whether the boundary length can extend infinitely (strip packing).
    infinite_length: bool,

    /// Cached area.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_area: Option<f64>,
}

impl Boundary2D {
    /// Creates a new boundary from polygon vertices.
    pub fn new(vertices: Vec<(f64, f64)>) -> Self {
        Self {
            exterior: vertices,
            holes: Vec::new(),
            width: None,
            height: None,
            infinite_length: false,
            cached_area: None,
        }
    }

    /// Creates a rectangular boundary.
    pub fn rectangle(width: f64, height: f64) -> Self {
        Self {
            exterior: vec![(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
            holes: Vec::new(),
            width: Some(width),
            height: Some(height),
            infinite_length: false,
            cached_area: Some(width * height),
        }
    }

    /// Creates an infinite strip (for strip packing problems).
    pub fn strip(width: f64) -> Self {
        Self {
            exterior: vec![(0.0, 0.0), (width, 0.0), (width, f64::MAX), (0.0, f64::MAX)],
            holes: Vec::new(),
            width: Some(width),
            height: None,
            infinite_length: true,
            cached_area: None,
        }
    }

    /// Adds an interior obstacle/hole.
    pub fn with_hole(mut self, vertices: Vec<(f64, f64)>) -> Self {
        self.holes.push(vertices);
        self.cached_area = None;
        self
    }

    /// Returns the width (if rectangular).
    pub fn width(&self) -> Option<f64> {
        self.width
    }

    /// Returns the height (if rectangular and not infinite).
    pub fn height(&self) -> Option<f64> {
        self.height
    }

    /// Returns whether this is an infinite strip.
    pub fn is_infinite(&self) -> bool {
        self.infinite_length
    }

    /// Returns the exterior vertices.
    pub fn exterior(&self) -> &[(f64, f64)] {
        &self.exterior
    }

    /// Returns the interior holes.
    pub fn holes(&self) -> &[Vec<(f64, f64)>] {
        &self.holes
    }

    /// Calculates the area of the boundary (exterior minus holes).
    fn calculate_area(&self) -> f64 {
        if self.infinite_length {
            return f64::INFINITY;
        }
        let mut total = geom_polygon::area(&self.exterior);
        for hole in &self.holes {
            total -= geom_polygon::area(hole);
        }
        total
    }
}

impl Boundary for Boundary2D {
    type Scalar = f64;

    fn measure(&self) -> f64 {
        if let Some(area) = self.cached_area {
            area
        } else {
            self.calculate_area()
        }
    }

    fn aabb(&self) -> ([f64; 2], [f64; 2]) {
        let (min, max) = self.aabb_vec();
        ([min[0], min[1]], [max[0], max[1]])
    }

    fn aabb_vec(&self) -> (Vec<f64>, Vec<f64>) {
        if self.exterior.is_empty() {
            return (vec![0.0, 0.0], vec![0.0, 0.0]);
        }

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for &(x, y) in &self.exterior {
            if y < f64::MAX / 2.0 {
                // Ignore infinite values
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }

        (vec![min_x, min_y], vec![max_x, max_y])
    }

    fn validate(&self) -> Result<()> {
        if self.exterior.len() < 3 {
            return Err(Error::InvalidBoundary(
                "Boundary must have at least 3 vertices".into(),
            ));
        }

        if let (Some(w), Some(h)) = (self.width, self.height) {
            if w <= 0.0 || h <= 0.0 {
                return Err(Error::InvalidBoundary(
                    "Width and height must be positive".into(),
                ));
            }
        }

        Ok(())
    }

    fn contains_point(&self, point: &[f64]) -> bool {
        if point.len() < 2 {
            return false;
        }
        let p = (point[0], point[1]);
        // Must be inside exterior and outside all holes
        if !geom_polygon::contains_point(&self.exterior, p) {
            return false;
        }
        for hole in &self.holes {
            if geom_polygon::contains_point(hole, p) {
                return false;
            }
        }
        true
    }
}

impl Boundary2DExt for Boundary2D {
    fn aabb_2d(&self) -> AABB2D<f64> {
        let (min, max) = self.aabb_vec();
        AABB2D::new(min[0], min[1], max[0], max[1])
    }

    fn vertices(&self) -> &[(f64, f64)] {
        &self.exterior
    }

    fn contains_polygon(&self, polygon: &[(f64, f64)]) -> bool {
        // Check if all vertices are inside the boundary
        for &p in polygon {
            if !geom_polygon::contains_point(&self.exterior, p) {
                return false;
            }
            // Also check that vertices are not inside any hole
            for hole in &self.holes {
                if geom_polygon::contains_point(hole, p) {
                    return false;
                }
            }
        }

        // For complete correctness, should also check edge intersections
        // but for performance, vertex containment is often sufficient
        true
    }

    fn effective_area(&self, margin: f64) -> f64 {
        if self.infinite_length {
            return f64::INFINITY;
        }

        if let (Some(w), Some(h)) = (self.width, self.height) {
            let eff_w = (w - 2.0 * margin).max(0.0);
            let eff_h = (h - 2.0 * margin).max(0.0);
            eff_w * eff_h
        } else {
            // For non-rectangular boundaries, approximate by subtracting perimeter * margin
            let perim = geom_polygon::perimeter(&self.exterior);
            (self.calculate_area() - perim * margin).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rectangle_boundary() {
        let boundary = Boundary2D::rectangle(100.0, 50.0);
        assert_relative_eq!(boundary.measure(), 5000.0, epsilon = 0.001);
        assert_eq!(boundary.width(), Some(100.0));
        assert_eq!(boundary.height(), Some(50.0));
    }

    #[test]
    fn test_strip_boundary() {
        let boundary = Boundary2D::strip(100.0);
        assert!(boundary.is_infinite());
        assert_eq!(boundary.width(), Some(100.0));
        assert_eq!(boundary.height(), None);
    }

    #[test]
    fn test_contains_point() {
        let boundary = Boundary2D::rectangle(100.0, 100.0);
        assert!(boundary.contains_point(&[50.0, 50.0]));
        assert!(!boundary.contains_point(&[150.0, 50.0]));
        assert!(!boundary.contains_point(&[-10.0, 50.0]));
    }

    #[test]
    fn test_effective_area() {
        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let eff = boundary.effective_area(10.0);
        // 80 * 80 = 6400
        assert_relative_eq!(eff, 6400.0, epsilon = 0.001);
    }

    #[test]
    fn test_aabb_2d() {
        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let aabb = boundary.aabb_2d();
        assert_relative_eq!(aabb.min_x, 0.0);
        assert_relative_eq!(aabb.min_y, 0.0);
        assert_relative_eq!(aabb.max_x, 100.0);
        assert_relative_eq!(aabb.max_y, 50.0);
    }

    #[test]
    fn test_validation() {
        let valid = Boundary2D::rectangle(100.0, 50.0);
        assert!(valid.validate().is_ok());

        let invalid = Boundary2D::new(vec![(0.0, 0.0), (1.0, 0.0)]);
        assert!(invalid.validate().is_err());
    }
}
