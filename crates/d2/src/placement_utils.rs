//! Shared utility functions for 2D nesting placement algorithms.
//!
//! This module consolidates common geometry operations used across
//! multiple nesting strategy implementations (GA, SA, BRKGA, ALNS, GDRR).

use crate::boundary::Boundary2D;
use crate::nfp::Nfp;
use u_nesting_core::geometry::Boundary;

/// Instance information for decoding placement orders.
///
/// Maps a flat instance index to the corresponding geometry and its
/// repetition number (when `quantity > 1`).
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// Index into the geometries array.
    pub geometry_idx: usize,
    /// Instance number within this geometry's quantity.
    pub instance_num: usize,
}

/// Computes the centroid (arithmetic mean of vertices) of a polygon.
///
/// # Returns
///
/// `(0.0, 0.0)` for an empty polygon, otherwise the mean of all vertices.
pub fn polygon_centroid(polygon: &[(f64, f64)]) -> (f64, f64) {
    if polygon.is_empty() {
        return (0.0, 0.0);
    }

    let sum: (f64, f64) = polygon
        .iter()
        .fold((0.0, 0.0), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let n = polygon.len() as f64;
    (sum.0 / n, sum.1 / n)
}

/// Grows an NFP by `spacing`, so that a reference point outside the result
/// keeps the two pieces at least `spacing` apart.
///
/// The NFP of pieces grown by `spacing / 2` each is the NFP grown by `spacing`
/// (the disc is symmetric), so this is the exact clearance region, computed as
/// a true polygon offset — see `polygon_ops::offset_polygon` for the accuracy
/// bound. Returns the NFP unchanged if `spacing <= 0.0`.
pub fn offset_nfp(nfp: &Nfp, spacing: f64) -> Nfp {
    if spacing <= 0.0 {
        return nfp.clone();
    }
    Nfp::from_polygons(
        nfp.polygons
            .iter()
            .flat_map(|polygon| crate::polygon_ops::offset_polygon(polygon, spacing))
            .collect(),
    )
}

/// The region pieces may occupy: the boundary moved inward by `margin`,
/// counter-clockwise.
///
/// A rectangle (or strip) is inset exactly on its bounding box. Any other
/// boundary is offset inward along its own edges — using its bounding box
/// instead would let a piece sit in the box's empty corners, closer than
/// `margin` to a slanted edge or outside the boundary altogether. If the
/// offset splits the boundary the largest piece is kept; if nothing is left,
/// the result is empty and nothing fits. Holes are not part of this region.
///
/// Placement code insets the boundary with this once and then computes the
/// inner-fit polygon with no further margin.
pub fn inset_boundary(boundary: &Boundary2D, margin: f64) -> Vec<(f64, f64)> {
    let rectangular =
        boundary.is_infinite() || (boundary.width().is_some() && boundary.height().is_some());
    if rectangular {
        let (b_min, b_max) = boundary.aabb();
        return vec![
            (b_min[0] + margin, b_min[1] + margin),
            (b_max[0] - margin, b_min[1] + margin),
            (b_max[0] - margin, b_max[1] - margin),
            (b_min[0] + margin, b_max[1] - margin),
        ];
    }
    let mut ring = crate::polygon_ops::offset_polygon(boundary.exterior(), -margin)
        .into_iter()
        .max_by(|a, b| {
            let area = |r: &[(f64, f64)]| crate::polygon_ops::signed_area2(r).abs();
            area(a).total_cmp(&area(b))
        })
        .unwrap_or_default();
    if crate::polygon_ops::signed_area2(&ring) < 0.0 {
        ring.reverse();
    }
    ring
}

/// Computes the nesting fitness score from placement results.
///
/// The fitness function prioritizes placement count (weight 100) over
/// utilization (weight 10), ensuring all-placed solutions always rank
/// higher than partial solutions.
///
/// # Arguments
///
/// * `placed_count` - Number of successfully placed instances
/// * `total_count` - Total number of instances to place
/// * `utilization` - Fraction of boundary area used (0.0..1.0)
///
/// # Returns
///
/// Fitness score in range [0, 110] where higher is better.
pub fn nesting_fitness(placed_count: usize, total_count: usize, utilization: f64) -> f64 {
    let placement_ratio = placed_count as f64 / total_count.max(1) as f64;
    placement_ratio * 100.0 + utilization * 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_centroid_empty() {
        assert_eq!(polygon_centroid(&[]), (0.0, 0.0));
    }

    #[test]
    fn test_polygon_centroid_square() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (cx, cy) = polygon_centroid(&square);
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn offset_nfp_with_no_spacing_is_the_nfp() {
        let nfp = Nfp::from_polygons(vec![vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ]]);
        assert_eq!(offset_nfp(&nfp, 0.0).polygons, nfp.polygons);
    }

    #[test]
    fn offset_nfp_moves_every_edge_by_the_spacing() {
        // An NFP whose corners sit on the diagonals: moving vertices away from
        // the centre shifted its edges by only 0.71 × spacing.
        let nfp = Nfp::from_polygons(vec![vec![
            (-300.0, -300.0),
            (300.0, -300.0),
            (300.0, 300.0),
            (-300.0, 300.0),
        ]]);
        let grown = offset_nfp(&nfp, 50.0);
        assert_eq!(grown.polygons.len(), 1);
        let ring = &grown.polygons[0];
        let max_x = ring.iter().map(|p| p.0).fold(f64::MIN, f64::max);
        let max_y = ring.iter().map(|p| p.1).fold(f64::MIN, f64::max);
        assert!((350.0..350.1).contains(&max_x), "right edge at {max_x}");
        assert!((350.0..350.1).contains(&max_y), "top edge at {max_y}");
    }

    #[test]
    fn inset_boundary_applies_the_margin_once_to_a_rectangle() {
        let rect = inset_boundary(&Boundary2D::rectangle(1000.0, 500.0), 50.0);
        assert_eq!(
            rect,
            vec![(50.0, 50.0), (950.0, 50.0), (950.0, 450.0), (50.0, 450.0)]
        );
    }

    #[test]
    fn inset_boundary_follows_a_slanted_edge() {
        // The bounding box of this triangle is 100 × 100; its hypotenuse is the
        // edge a box inset would ignore.
        let triangle = Boundary2D::new(vec![(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]);
        let ring = inset_boundary(&triangle, 10.0);
        assert!(ring.len() >= 3);
        for &(x, y) in &ring {
            let to_hypotenuse = (100.0 - x - y) / 2f64.sqrt();
            assert!(x >= 10.0 - 1e-6 && y >= 10.0 - 1e-6 && to_hypotenuse >= 10.0 - 1e-6);
        }
    }

    #[test]
    fn test_nesting_fitness_all_placed() {
        let f = nesting_fitness(10, 10, 0.85);
        assert!((f - 108.5).abs() < 1e-10);
    }

    #[test]
    fn test_nesting_fitness_partial() {
        let f = nesting_fitness(5, 10, 0.40);
        // 0.5 * 100 + 0.4 * 10 = 54.0
        assert!((f - 54.0).abs() < 1e-10);
    }

    #[test]
    fn test_nesting_fitness_none_placed() {
        let f = nesting_fitness(0, 10, 0.0);
        assert!((f - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nesting_fitness_empty_total() {
        let f = nesting_fitness(0, 0, 0.0);
        assert!((f - 0.0).abs() < 1e-10);
    }
}
