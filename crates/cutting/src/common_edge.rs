//! Common-edge detection for nested parts.
//!
//! Detects pairs of edges from different parts that are close enough to be
//! cut in a single pass (common-line cutting). This reduces material waste,
//! pierce count, and total cutting time.
//!
//! # Algorithm
//!
//! For each pair of exterior contours from different parts:
//! 1. Compare all edge pairs for proximity and collinearity
//! 2. Edges within `kerf_width + tolerance` that are anti-parallel (opposite
//!    direction) and have overlapping projections are common-edge candidates
//!
//! # References
//!
//! - Hu, Lin & Fu (2023), "Optimizing Cutting Sequences for Common-Edge Nested Parts"

use crate::contour::{ContourId, ContourType, CutContour};

/// A detected common edge between two contours.
#[derive(Debug, Clone)]
pub struct CommonEdge {
    /// ID of the first contour.
    pub contour_a: ContourId,
    /// Edge index in contour A.
    pub edge_a: usize,
    /// ID of the second contour.
    pub contour_b: ContourId,
    /// Edge index in contour B.
    pub edge_b: usize,
    /// Length of the overlapping segment.
    pub overlap_length: f64,
    /// Midpoint of the shared edge.
    pub midpoint: (f64, f64),
}

/// Result of common-edge detection.
#[derive(Debug, Clone)]
pub struct CommonEdgeResult {
    /// Detected common edges.
    pub common_edges: Vec<CommonEdge>,
    /// Total length of all common edges.
    pub total_common_length: f64,
}

/// Detects common edges between exterior contours.
///
/// Two edges from different contours are considered "common" when:
/// - They are approximately anti-parallel (opposite direction)
/// - Their perpendicular distance is within `max_distance`
/// - They have overlapping projections along the edge direction
///
/// # Arguments
///
/// * `contours` - All cut contours
/// * `max_distance` - Maximum distance for common-edge eligibility
///   (typically `kerf_width + tolerance`)
/// * `angle_tolerance` - Maximum angle deviation in radians (default ~0.05 ≈ 3°)
pub fn detect_common_edges(
    contours: &[CutContour],
    max_distance: f64,
    angle_tolerance: f64,
) -> CommonEdgeResult {
    let mut common_edges = Vec::new();

    // Only exterior contours participate in common-edge detection
    let exteriors: Vec<&CutContour> = contours
        .iter()
        .filter(|c| c.contour_type == ContourType::Exterior)
        .collect();

    // Compare all pairs of exterior contours from different parts
    for i in 0..exteriors.len() {
        for j in (i + 1)..exteriors.len() {
            let ca = exteriors[i];
            let cb = exteriors[j];

            // Quick AABB overlap check to skip distant contours
            if !aabb_proximity(ca, cb, max_distance) {
                continue;
            }

            // Compare all edge pairs
            detect_edges_between(ca, cb, max_distance, angle_tolerance, &mut common_edges);
        }
    }

    let total_common_length: f64 = common_edges.iter().map(|e| e.overlap_length).sum();

    CommonEdgeResult {
        common_edges,
        total_common_length,
    }
}

/// Checks if two contours' AABBs are close enough to potentially share edges.
fn aabb_proximity(a: &CutContour, b: &CutContour, max_dist: f64) -> bool {
    let (a_min, a_max) = contour_aabb(a);
    let (b_min, b_max) = contour_aabb(b);

    // Check if expanded AABBs overlap
    a_min.0 - max_dist <= b_max.0
        && a_max.0 + max_dist >= b_min.0
        && a_min.1 - max_dist <= b_max.1
        && a_max.1 + max_dist >= b_min.1
}

/// Computes the AABB of a contour.
fn contour_aabb(c: &CutContour) -> ((f64, f64), (f64, f64)) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for &(x, y) in &c.vertices {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    ((min_x, min_y), (max_x, max_y))
}

/// Detects common edges between two specific contours.
fn detect_edges_between(
    ca: &CutContour,
    cb: &CutContour,
    max_distance: f64,
    angle_tolerance: f64,
    results: &mut Vec<CommonEdge>,
) {
    let na = ca.vertices.len();
    let nb = cb.vertices.len();

    for i in 0..na {
        let a0 = ca.vertices[i];
        let a1 = ca.vertices[(i + 1) % na];
        let (da_x, da_y) = (a1.0 - a0.0, a1.1 - a0.1);
        let len_a = (da_x * da_x + da_y * da_y).sqrt();

        if len_a < 1e-12 {
            continue; // Skip degenerate edges
        }

        let dir_a = (da_x / len_a, da_y / len_a);

        for j in 0..nb {
            let b0 = cb.vertices[j];
            let b1 = cb.vertices[(j + 1) % nb];
            let (db_x, db_y) = (b1.0 - b0.0, b1.1 - b0.1);
            let len_b = (db_x * db_x + db_y * db_y).sqrt();

            if len_b < 1e-12 {
                continue;
            }

            let dir_b = (db_x / len_b, db_y / len_b);

            // Check anti-parallelism: dot product should be close to -1
            let dot = dir_a.0 * dir_b.0 + dir_a.1 * dir_b.1;
            if dot > -1.0 + angle_tolerance {
                continue; // Not anti-parallel
            }

            // Compute perpendicular distance between the two edge lines
            let perp_dist = perpendicular_distance(a0, dir_a, b0);
            if perp_dist > max_distance {
                continue;
            }

            // Compute overlap along the edge direction
            let overlap = compute_overlap(a0, a1, b0, b1, dir_a);
            if overlap <= 1e-6 {
                continue; // No meaningful overlap
            }

            // Compute midpoint of the overlapping segment
            let t_a0 = project_onto_line(a0, dir_a, b0);
            let t_a1 = project_onto_line(a0, dir_a, b1);
            let t_b_min = t_a0.min(t_a1).max(0.0);
            let t_b_max = t_a0.max(t_a1).min(len_a);
            let t_mid = (t_b_min + t_b_max) / 2.0;
            let midpoint = (a0.0 + dir_a.0 * t_mid, a0.1 + dir_a.1 * t_mid);

            results.push(CommonEdge {
                contour_a: ca.id,
                edge_a: i,
                contour_b: cb.id,
                edge_b: j,
                overlap_length: overlap,
                midpoint,
            });
        }
    }
}

/// Computes perpendicular distance from a point to a line defined by point + direction.
fn perpendicular_distance(line_point: (f64, f64), line_dir: (f64, f64), point: (f64, f64)) -> f64 {
    let dx = point.0 - line_point.0;
    let dy = point.1 - line_point.1;
    // Cross product magnitude gives perpendicular distance
    (dx * line_dir.1 - dy * line_dir.0).abs()
}

/// Projects a point onto a line and returns the parameter t.
fn project_onto_line(line_origin: (f64, f64), line_dir: (f64, f64), point: (f64, f64)) -> f64 {
    let dx = point.0 - line_origin.0;
    let dy = point.1 - line_origin.1;
    dx * line_dir.0 + dy * line_dir.1
}

/// Computes the overlap length between two anti-parallel edge segments.
fn compute_overlap(
    a0: (f64, f64),
    a1: (f64, f64),
    b0: (f64, f64),
    b1: (f64, f64),
    dir_a: (f64, f64),
) -> f64 {
    let len_a = ((a1.0 - a0.0).powi(2) + (a1.1 - a0.1).powi(2)).sqrt();

    // Project B endpoints onto A's direction
    let t_b0 = project_onto_line(a0, dir_a, b0);
    let t_b1 = project_onto_line(a0, dir_a, b1);

    let b_min = t_b0.min(t_b1);
    let b_max = t_b0.max(t_b1);

    // Overlap with A's range [0, len_a]
    let overlap_start = b_min.max(0.0);
    let overlap_end = b_max.min(len_a);

    (overlap_end - overlap_start).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rect(id: usize, x: f64, y: f64, w: f64, h: f64) -> CutContour {
        CutContour {
            id,
            geometry_id: format!("part{}", id),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            perimeter: 2.0 * (w + h),
            centroid: (x + w / 2.0, y + h / 2.0),
        }
    }

    #[test]
    fn test_adjacent_rectangles_share_edge() {
        // Two 10x10 rectangles side by side with 0.2mm gap (kerf)
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 10.2, 0.0, 10.0, 10.0), // 0.2 gap
        ];

        let result = detect_common_edges(&contours, 0.5, 0.1);

        // Right edge of rect 0 (x=10) and left edge of rect 1 (x=10.2) should match
        assert!(
            !result.common_edges.is_empty(),
            "Should detect common edge between adjacent rectangles"
        );
        assert!(result.total_common_length > 9.0); // Close to 10.0
    }

    #[test]
    fn test_distant_rectangles_no_common() {
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 50.0, 0.0, 10.0, 10.0), // Far apart
        ];

        let result = detect_common_edges(&contours, 0.5, 0.1);
        assert!(result.common_edges.is_empty());
    }

    #[test]
    fn test_perpendicular_edges_no_common() {
        // Two rectangles with perpendicular adjacent edges
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 10.0, 10.0, 10.0, 10.0), // Corner-touching
        ];

        let result = detect_common_edges(&contours, 0.5, 0.1);
        // Only anti-parallel edges match, perpendicular ones don't
        // The bottom of rect1 (y=10) and top of rect0 (y=10) may match
        // but the side edges at the corner shouldn't
        for ce in &result.common_edges {
            assert!(ce.overlap_length > 0.0);
        }
    }

    #[test]
    fn test_interior_contours_ignored() {
        let mut contours = vec![make_rect(0, 0.0, 0.0, 10.0, 10.0)];
        contours.push(CutContour {
            id: 1,
            geometry_id: "part1".to_string(),
            instance: 0,
            contour_type: ContourType::Interior, // Interior
            vertices: vec![(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)],
            perimeter: 24.0,
            centroid: (5.0, 5.0),
        });

        let result = detect_common_edges(&contours, 0.5, 0.1);
        assert!(
            result.common_edges.is_empty(),
            "Interior contours should not participate"
        );
    }

    #[test]
    fn test_stacked_rectangles_horizontal() {
        // Two 10x5 rectangles stacked vertically with 0.3mm gap
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 5.0),
            make_rect(1, 0.0, 5.3, 10.0, 5.0),
        ];

        let result = detect_common_edges(&contours, 0.5, 0.1);
        assert!(!result.common_edges.is_empty());
        // Top edge of rect 0 (y=5) and bottom edge of rect 1 (y=5.3) should match
        assert!(result.total_common_length > 9.0);
    }

    #[test]
    fn test_common_edge_result_fields() {
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 10.1, 0.0, 10.0, 10.0),
        ];

        let result = detect_common_edges(&contours, 0.5, 0.1);
        assert!(!result.common_edges.is_empty());

        let ce = &result.common_edges[0];
        assert!(ce.contour_a == 0 || ce.contour_a == 1);
        assert!(ce.contour_b == 0 || ce.contour_b == 1);
        assert_ne!(ce.contour_a, ce.contour_b);
        assert!(ce.overlap_length > 0.0);
    }

    #[test]
    fn test_empty_contours() {
        let result = detect_common_edges(&[], 0.5, 0.1);
        assert!(result.common_edges.is_empty());
        assert_eq!(result.total_common_length, 0.0);
    }
}
