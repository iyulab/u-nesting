//! Robust geometric predicates for numerical stability.
//!
//! This module provides numerically robust implementations of geometric predicates
//! using Shewchuk's adaptive precision floating-point arithmetic.
//!
//! ## Core Predicates
//!
//! The core predicates (`orient2d`, `orient2d_filtered`, `point_in_triangle`,
//! `is_convex`, `is_ccw`, `Orientation`) are provided by `u-geometry` and
//! re-exported here for API compatibility.
//!
//! ## Nesting-Specific Utilities
//!
//! - [`signed_area_robust`]: Signed polygon area with Kahan summation for numerical stability
//! - [`ScalingConfig`]: Coordinate scaling for integer-range robust arithmetic
//! - [`snap_to_grid`] / [`snap_polygon_to_grid`]: Grid-snapping utilities
//!
//! ## References
//!
//! - Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic and
//!   Fast Robust Predicates for Computational Geometry"
//! - <https://www.cs.cmu.edu/~quake/robust.html>
//!
//! ## Example
//!
//! ```rust
//! use u_nesting_core::robust::{orient2d, Orientation};
//!
//! // Check orientation of three points
//! let a = (0.0, 0.0);
//! let b = (1.0, 0.0);
//! let c = (0.5, 1.0);
//!
//! match orient2d(a, b, c) {
//!     Orientation::CounterClockwise => println!("Left turn"),
//!     Orientation::Clockwise => println!("Right turn"),
//!     Orientation::Collinear => println!("Straight"),
//! }
//! ```

// ============================================================================
// Re-exports from u-geometry (canonical implementations)
// ============================================================================

pub use u_geometry::robust::{
    is_ccw, is_convex, orient2d, orient2d_filtered, orient2d_raw, point_in_triangle,
    point_in_triangle_inclusive, Orientation,
};

// ============================================================================
// Nesting-specific aliases (backward compatibility)
// ============================================================================

/// Checks if a point lies strictly inside a triangle.
///
/// This is an alias for [`point_in_triangle`] preserved for backward compatibility
/// with existing u-nesting code that uses the `_robust` suffix convention.
#[inline]
pub fn point_in_triangle_robust(
    p: (f64, f64),
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> bool {
    point_in_triangle(p, a, b, c)
}

/// Checks if a point lies inside or on the boundary of a triangle.
///
/// This is an alias for [`point_in_triangle_inclusive`] preserved for backward
/// compatibility with existing u-nesting code.
#[inline]
pub fn point_in_triangle_inclusive_robust(
    p: (f64, f64),
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> bool {
    point_in_triangle_inclusive(p, a, b, c)
}

/// Checks if a polygon is convex using robust orientation tests.
///
/// This is an alias for [`is_convex`] preserved for backward compatibility.
#[inline]
pub fn is_convex_robust(polygon: &[(f64, f64)]) -> bool {
    is_convex(polygon)
}

/// Checks if a polygon has counter-clockwise winding order.
///
/// This is an alias for [`is_ccw`] preserved for backward compatibility.
#[inline]
pub fn is_ccw_robust(polygon: &[(f64, f64)]) -> bool {
    is_ccw(polygon)
}

// ============================================================================
// Nesting-Specific Utilities (not in u-geometry)
// ============================================================================

/// Computes the signed area of a polygon using Kahan summation.
///
/// The shoelace formula is used with Kahan compensated summation for
/// better numerical stability on large polygons with diverse vertex scales.
///
/// # Arguments
///
/// * `polygon` - Slice of polygon vertices in order
///
/// # Returns
///
/// Positive area if counter-clockwise, negative if clockwise
///
/// # Reference
/// Kahan (1965), "Pracniques: Further Remarks on Reducing Truncation Errors"
pub fn signed_area_robust(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }

    // Kahan summation for better numerical stability
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for i in 0..n {
        let j = (i + 1) % n;
        let term = polygon[i].0 * polygon[j].1 - polygon[j].0 * polygon[i].1;

        let y = term - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum / 2.0
}

// ============================================================================
// Integer Coordinate Scaling
// ============================================================================

/// Configuration for coordinate scaling.
#[derive(Debug, Clone, Copy)]
pub struct ScalingConfig {
    /// The scale factor (coordinates are multiplied by this value).
    pub scale: f64,
    /// The inverse scale factor (for converting back).
    pub inv_scale: f64,
}

impl ScalingConfig {
    /// Creates a new scaling configuration.
    ///
    /// # Arguments
    ///
    /// * `precision` - Number of decimal places to preserve
    ///
    /// # Example
    ///
    /// ```rust
    /// use u_nesting_core::robust::ScalingConfig;
    ///
    /// // Preserve 3 decimal places
    /// let config = ScalingConfig::new(3);
    /// assert_eq!(config.scale, 1000.0);
    /// ```
    pub fn new(precision: u32) -> Self {
        let scale = 10.0_f64.powi(precision as i32);
        Self {
            scale,
            inv_scale: 1.0 / scale,
        }
    }

    /// Scales a coordinate to integer range.
    #[inline]
    pub fn scale_coord(&self, x: f64) -> f64 {
        (x * self.scale).round()
    }

    /// Scales a point to integer range.
    #[inline]
    pub fn scale_point(&self, p: (f64, f64)) -> (f64, f64) {
        (self.scale_coord(p.0), self.scale_coord(p.1))
    }

    /// Unscales a coordinate back to original range.
    #[inline]
    pub fn unscale_coord(&self, x: f64) -> f64 {
        x * self.inv_scale
    }

    /// Unscales a point back to original range.
    #[inline]
    pub fn unscale_point(&self, p: (f64, f64)) -> (f64, f64) {
        (self.unscale_coord(p.0), self.unscale_coord(p.1))
    }

    /// Scales an entire polygon.
    pub fn scale_polygon(&self, polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
        polygon.iter().map(|&p| self.scale_point(p)).collect()
    }

    /// Unscales an entire polygon.
    pub fn unscale_polygon(&self, polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
        polygon.iter().map(|&p| self.unscale_point(p)).collect()
    }
}

impl Default for ScalingConfig {
    /// Default scaling preserves 6 decimal places.
    fn default() -> Self {
        Self::new(6)
    }
}

/// Snaps coordinates to a grid of the given resolution.
///
/// # Arguments
///
/// * `point` - The point to snap
/// * `resolution` - Grid cell size
///
/// # Returns
///
/// The point snapped to the nearest grid intersection
#[inline]
pub fn snap_to_grid(point: (f64, f64), resolution: f64) -> (f64, f64) {
    (
        (point.0 / resolution).round() * resolution,
        (point.1 / resolution).round() * resolution,
    )
}

/// Snaps an entire polygon to a grid.
pub fn snap_polygon_to_grid(polygon: &[(f64, f64)], resolution: f64) -> Vec<(f64, f64)> {
    polygon
        .iter()
        .map(|&p| snap_to_grid(p, resolution))
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient2d_basic() {
        // Counter-clockwise triangle
        let a = (0.0, 0.0);
        let b = (1.0, 0.0);
        let c = (0.5, 1.0);

        assert_eq!(orient2d(a, b, c), Orientation::CounterClockwise);
        assert_eq!(orient2d(a, c, b), Orientation::Clockwise);
    }

    #[test]
    fn test_orient2d_collinear() {
        let a = (0.0, 0.0);
        let b = (1.0, 1.0);
        let c = (2.0, 2.0);

        assert_eq!(orient2d(a, b, c), Orientation::Collinear);
    }

    #[test]
    fn test_orient2d_near_collinear() {
        let a = (0.0, 0.0);
        let b = (1.0, 1.0);
        let c = (2.0, 2.0 + 1e-15);

        let result = orient2d(a, b, c);
        assert!(
            result == Orientation::Collinear || result == Orientation::CounterClockwise,
            "Expected collinear or CCW, got {:?}",
            result
        );
    }

    #[test]
    fn test_orient2d_filtered_fast_path() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);

        assert_eq!(orient2d_filtered(a, b, c), Orientation::CounterClockwise);
    }

    #[test]
    fn test_point_in_triangle_robust() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);

        assert!(point_in_triangle_robust((5.0, 3.0), a, b, c));
        assert!(!point_in_triangle_robust((20.0, 5.0), a, b, c));
        assert!(!point_in_triangle_robust((5.0, 0.0), a, b, c));
    }

    #[test]
    fn test_point_in_triangle_inclusive() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);

        assert!(point_in_triangle_inclusive_robust((5.0, 3.0), a, b, c));
        assert!(point_in_triangle_inclusive_robust((5.0, 0.0), a, b, c));
        assert!(point_in_triangle_inclusive_robust((0.0, 0.0), a, b, c));
        assert!(!point_in_triangle_inclusive_robust((20.0, 5.0), a, b, c));
    }

    #[test]
    fn test_is_convex_robust() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_convex_robust(&square));

        let triangle = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!(is_convex_robust(&triangle));

        let l_shape = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (5.0, 5.0),
            (5.0, 10.0),
            (0.0, 10.0),
        ];
        assert!(!is_convex_robust(&l_shape));
    }

    #[test]
    fn test_is_ccw_robust() {
        let ccw_square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_ccw_robust(&ccw_square));

        let cw_square = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        assert!(!is_ccw_robust(&cw_square));
    }

    #[test]
    fn test_signed_area_robust() {
        let ccw_square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let area = signed_area_robust(&ccw_square);
        assert!((area - 100.0).abs() < 1e-10);

        let cw_square = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let area = signed_area_robust(&cw_square);
        assert!((area + 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_config() {
        let config = ScalingConfig::new(3);

        let p = (1.234, 5.678);
        let scaled = config.scale_point(p);
        assert_eq!(scaled, (1234.0, 5678.0));

        let unscaled = config.unscale_point(scaled);
        assert!((unscaled.0 - p.0).abs() < 1e-10);
        assert!((unscaled.1 - p.1).abs() < 1e-10);
    }

    #[test]
    fn test_snap_to_grid() {
        let p = (1.23, 4.56);

        let snapped = snap_to_grid(p, 0.5);
        assert_eq!(snapped, (1.0, 4.5));

        let snapped = snap_to_grid(p, 1.0);
        assert_eq!(snapped, (1.0, 5.0));
    }

    #[test]
    fn test_orientation_methods() {
        assert!(Orientation::CounterClockwise.is_ccw());
        assert!(!Orientation::CounterClockwise.is_cw());
        assert!(!Orientation::CounterClockwise.is_collinear());

        assert!(!Orientation::Clockwise.is_ccw());
        assert!(Orientation::Clockwise.is_cw());
        assert!(!Orientation::Clockwise.is_collinear());

        assert!(!Orientation::Collinear.is_ccw());
        assert!(!Orientation::Collinear.is_cw());
        assert!(Orientation::Collinear.is_collinear());
    }

    #[test]
    fn test_degenerate_triangle() {
        let a = (0.0, 0.0);
        let b = (5.0, 0.0);
        let c = (10.0, 0.0);

        assert!(!point_in_triangle_robust((5.0, 0.0), a, b, c));
    }

    #[test]
    fn test_extreme_coordinates() {
        let a = (1e10, 1e10);
        let b = (1e10 + 1.0, 1e10);
        let c = (1e10 + 0.5, 1e10 + 1.0);

        assert_eq!(orient2d(a, b, c), Orientation::CounterClockwise);

        let a = (1e-10, 1e-10);
        let b = (1e-10 + 1e-12, 1e-10);
        let c = (1e-10 + 5e-13, 1e-10 + 1e-12);

        let result = orient2d(a, b, c);
        assert!(
            result == Orientation::CounterClockwise || result == Orientation::Collinear,
            "Unexpected orientation: {:?}",
            result
        );
    }
}
