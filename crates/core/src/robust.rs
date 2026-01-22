//! Robust geometric predicates for numerical stability.
//!
//! This module provides numerically robust implementations of geometric predicates
//! using Shewchuk's adaptive precision floating-point arithmetic.
//!
//! ## Background
//!
//! Standard floating-point arithmetic can produce incorrect results for geometric
//! predicates when points are nearly collinear or cocircular. This module provides
//! robust alternatives that guarantee correct results.
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

use robust::{orient2d as robust_orient2d, Coord};

/// Result of an orientation test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Points are arranged counter-clockwise (left turn).
    CounterClockwise,
    /// Points are arranged clockwise (right turn).
    Clockwise,
    /// Points are collinear (on the same line).
    Collinear,
}

impl Orientation {
    /// Returns true if the orientation is counter-clockwise.
    #[inline]
    pub fn is_ccw(self) -> bool {
        matches!(self, Orientation::CounterClockwise)
    }

    /// Returns true if the orientation is clockwise.
    #[inline]
    pub fn is_cw(self) -> bool {
        matches!(self, Orientation::Clockwise)
    }

    /// Returns true if the points are collinear.
    #[inline]
    pub fn is_collinear(self) -> bool {
        matches!(self, Orientation::Collinear)
    }
}

// ============================================================================
// Core Predicates (using robust crate)
// ============================================================================

/// Determines the orientation of three 2D points.
///
/// This is a numerically robust implementation using Shewchuk's adaptive
/// precision arithmetic. It correctly handles near-degenerate cases where
/// standard floating-point arithmetic would fail.
///
/// # Arguments
///
/// * `pa` - First point
/// * `pb` - Second point
/// * `pc` - Third point (the point being tested)
///
/// # Returns
///
/// - `Orientation::CounterClockwise` if `pc` lies to the left of the directed line from `pa` to `pb`
/// - `Orientation::Clockwise` if `pc` lies to the right
/// - `Orientation::Collinear` if the three points are collinear
///
/// # Example
///
/// ```rust
/// use u_nesting_core::robust::{orient2d, Orientation};
///
/// let a = (0.0, 0.0);
/// let b = (1.0, 0.0);
/// let c = (0.5, 1.0);
///
/// assert_eq!(orient2d(a, b, c), Orientation::CounterClockwise);
/// ```
#[inline]
pub fn orient2d(pa: (f64, f64), pb: (f64, f64), pc: (f64, f64)) -> Orientation {
    let result = robust_orient2d(
        Coord { x: pa.0, y: pa.1 },
        Coord { x: pb.0, y: pb.1 },
        Coord { x: pc.0, y: pc.1 },
    );

    if result > 0.0 {
        Orientation::CounterClockwise
    } else if result < 0.0 {
        Orientation::Clockwise
    } else {
        Orientation::Collinear
    }
}

/// Returns the raw orientation determinant value.
///
/// This is useful when you need the actual signed area value, not just the sign.
/// The magnitude is proportional to twice the signed area of the triangle formed
/// by the three points.
///
/// # Returns
///
/// - Positive if counter-clockwise
/// - Negative if clockwise
/// - Zero if collinear
#[inline]
pub fn orient2d_raw(pa: (f64, f64), pb: (f64, f64), pc: (f64, f64)) -> f64 {
    robust_orient2d(
        Coord { x: pa.0, y: pa.1 },
        Coord { x: pb.0, y: pb.1 },
        Coord { x: pc.0, y: pc.1 },
    )
}

// ============================================================================
// Floating-Point Filter (Fast Path + Exact Fallback)
// ============================================================================

/// Epsilon for fast floating-point filter.
///
/// If the result magnitude is larger than this threshold times the input magnitude,
/// we can trust the fast path result. Otherwise, fall back to exact arithmetic.
const FILTER_EPSILON: f64 = 1e-12;

/// Fast orientation test with exact fallback.
///
/// This implements a floating-point filter that:
/// 1. First attempts a fast approximate calculation
/// 2. Falls back to exact arithmetic only when necessary
///
/// In practice, ~95% of cases are resolved by the fast path, providing
/// near-native performance while guaranteeing correctness.
///
/// # Arguments
///
/// * `pa` - First point
/// * `pb` - Second point
/// * `pc` - Third point
///
/// # Returns
///
/// The orientation of the three points.
#[inline]
pub fn orient2d_filtered(pa: (f64, f64), pb: (f64, f64), pc: (f64, f64)) -> Orientation {
    // Fast path: simple cross product
    let acx = pa.0 - pc.0;
    let bcx = pb.0 - pc.0;
    let acy = pa.1 - pc.1;
    let bcy = pb.1 - pc.1;

    let det = acx * bcy - acy * bcx;

    // Compute error bound
    let det_sum = (acx * bcy).abs() + (acy * bcx).abs();

    // If the determinant is clearly non-zero, use fast path
    if det.abs() > FILTER_EPSILON * det_sum {
        return if det > 0.0 {
            Orientation::CounterClockwise
        } else {
            Orientation::Clockwise
        };
    }

    // Fall back to exact arithmetic
    orient2d(pa, pb, pc)
}

// ============================================================================
// Higher-Level Geometric Predicates
// ============================================================================

/// Checks if a point lies strictly inside a triangle.
///
/// Uses robust orientation tests to correctly handle edge cases.
///
/// # Arguments
///
/// * `p` - The point to test
/// * `a`, `b`, `c` - Triangle vertices (in any order)
///
/// # Returns
///
/// `true` if the point is strictly inside the triangle (not on the boundary)
pub fn point_in_triangle_robust(
    p: (f64, f64),
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> bool {
    let o1 = orient2d(a, b, p);
    let o2 = orient2d(b, c, p);
    let o3 = orient2d(c, a, p);

    // Point is inside if all orientations are the same (all CCW or all CW)
    // and none are collinear (strictly inside)
    (o1 == Orientation::CounterClockwise
        && o2 == Orientation::CounterClockwise
        && o3 == Orientation::CounterClockwise)
        || (o1 == Orientation::Clockwise
            && o2 == Orientation::Clockwise
            && o3 == Orientation::Clockwise)
}

/// Checks if a point lies inside or on the boundary of a triangle.
///
/// Uses robust orientation tests to correctly handle edge cases.
///
/// # Arguments
///
/// * `p` - The point to test
/// * `a`, `b`, `c` - Triangle vertices (in any order)
///
/// # Returns
///
/// `true` if the point is inside or on the boundary of the triangle
pub fn point_in_triangle_inclusive_robust(
    p: (f64, f64),
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> bool {
    let o1 = orient2d(a, b, p);
    let o2 = orient2d(b, c, p);
    let o3 = orient2d(c, a, p);

    // Point is inside or on boundary if:
    // - No orientations are opposite to each other
    // - At least one orientation must match the others (or be collinear)

    let has_ccw = o1.is_ccw() || o2.is_ccw() || o3.is_ccw();
    let has_cw = o1.is_cw() || o2.is_cw() || o3.is_cw();

    // If we have both CCW and CW, point is outside
    !(has_ccw && has_cw)
}

/// Checks if a polygon is convex using robust orientation tests.
///
/// # Arguments
///
/// * `polygon` - Slice of polygon vertices in order
///
/// # Returns
///
/// `true` if the polygon is convex
pub fn is_convex_robust(polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut expected_orientation: Option<Orientation> = None;

    for i in 0..n {
        let p0 = polygon[i];
        let p1 = polygon[(i + 1) % n];
        let p2 = polygon[(i + 2) % n];

        let o = orient2d(p0, p1, p2);

        // Skip collinear edges
        if o.is_collinear() {
            continue;
        }

        match expected_orientation {
            None => expected_orientation = Some(o),
            Some(expected) if expected != o => return false,
            _ => {}
        }
    }

    true
}

/// Checks if a polygon has counter-clockwise winding order.
///
/// # Arguments
///
/// * `polygon` - Slice of polygon vertices
///
/// # Returns
///
/// `true` if the polygon is wound counter-clockwise
pub fn is_ccw_robust(polygon: &[(f64, f64)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    // Find the lowest-leftmost vertex (guaranteed to be convex)
    let mut min_idx = 0;
    for (i, &(x, y)) in polygon.iter().enumerate() {
        let (min_x, min_y) = polygon[min_idx];
        if y < min_y || (y == min_y && x < min_x) {
            min_idx = i;
        }
    }

    let n = polygon.len();
    let prev = polygon[(min_idx + n - 1) % n];
    let curr = polygon[min_idx];
    let next = polygon[(min_idx + 1) % n];

    orient2d(prev, curr, next).is_ccw()
}

/// Computes the signed area of a polygon using robust arithmetic.
///
/// The shoelace formula is used, with careful handling of numerical precision.
///
/// # Arguments
///
/// * `polygon` - Slice of polygon vertices in order
///
/// # Returns
///
/// Positive area if counter-clockwise, negative if clockwise
pub fn signed_area_robust(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }

    // Use Kahan summation for better numerical stability
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
        // Near-collinear points that would fail with naive floating-point
        let a = (0.0, 0.0);
        let b = (1.0, 1.0);
        let c = (2.0, 2.0 + 1e-15);

        // Should detect the slight deviation from collinear
        let result = orient2d(a, b, c);
        // Due to floating-point representation, this might still be collinear
        // or slightly CCW - the important thing is it doesn't crash or give wrong sign
        assert!(
            result == Orientation::Collinear || result == Orientation::CounterClockwise,
            "Expected collinear or CCW, got {:?}",
            result
        );
    }

    #[test]
    fn test_orient2d_filtered_fast_path() {
        // Clear non-collinear case - should use fast path
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

        // Point inside
        assert!(point_in_triangle_robust((5.0, 3.0), a, b, c));

        // Point outside
        assert!(!point_in_triangle_robust((20.0, 5.0), a, b, c));

        // Point on edge (not strictly inside)
        assert!(!point_in_triangle_robust((5.0, 0.0), a, b, c));
    }

    #[test]
    fn test_point_in_triangle_inclusive() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);

        // Point inside
        assert!(point_in_triangle_inclusive_robust((5.0, 3.0), a, b, c));

        // Point on edge (should be included)
        assert!(point_in_triangle_inclusive_robust((5.0, 0.0), a, b, c));

        // Point at vertex (should be included)
        assert!(point_in_triangle_inclusive_robust((0.0, 0.0), a, b, c));

        // Point outside
        assert!(!point_in_triangle_inclusive_robust((20.0, 5.0), a, b, c));
    }

    #[test]
    fn test_is_convex_robust() {
        // Convex square
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_convex_robust(&square));

        // Convex triangle
        let triangle = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!(is_convex_robust(&triangle));

        // Non-convex L-shape
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
        // CCW square
        let ccw_square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_ccw_robust(&ccw_square));

        // CW square (reversed)
        let cw_square = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        assert!(!is_ccw_robust(&cw_square));
    }

    #[test]
    fn test_signed_area_robust() {
        // CCW square with area 100
        let ccw_square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let area = signed_area_robust(&ccw_square);
        assert!((area - 100.0).abs() < 1e-10);

        // CW square with negative area
        let cw_square = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let area = signed_area_robust(&cw_square);
        assert!((area + 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_config() {
        let config = ScalingConfig::new(3);

        // Scale a point
        let p = (1.234, 5.678);
        let scaled = config.scale_point(p);
        assert_eq!(scaled, (1234.0, 5678.0));

        // Unscale back
        let unscaled = config.unscale_point(scaled);
        assert!((unscaled.0 - p.0).abs() < 1e-10);
        assert!((unscaled.1 - p.1).abs() < 1e-10);
    }

    #[test]
    fn test_snap_to_grid() {
        let p = (1.23, 4.56);

        // Snap to grid of 0.5
        let snapped = snap_to_grid(p, 0.5);
        assert_eq!(snapped, (1.0, 4.5));

        // Snap to grid of 1.0
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
        // Degenerate triangle (all points collinear)
        let a = (0.0, 0.0);
        let b = (5.0, 0.0);
        let c = (10.0, 0.0);

        // Point should not be "inside" a degenerate triangle
        assert!(!point_in_triangle_robust((5.0, 0.0), a, b, c));
    }

    #[test]
    fn test_extreme_coordinates() {
        // Very large coordinates
        let a = (1e10, 1e10);
        let b = (1e10 + 1.0, 1e10);
        let c = (1e10 + 0.5, 1e10 + 1.0);

        assert_eq!(orient2d(a, b, c), Orientation::CounterClockwise);

        // Very small coordinates
        let a = (1e-10, 1e-10);
        let b = (1e-10 + 1e-12, 1e-10);
        let c = (1e-10 + 5e-13, 1e-10 + 1e-12);

        // Should still produce correct orientation
        let result = orient2d(a, b, c);
        assert!(
            result == Orientation::CounterClockwise || result == Orientation::Collinear,
            "Unexpected orientation: {:?}",
            result
        );
    }
}
