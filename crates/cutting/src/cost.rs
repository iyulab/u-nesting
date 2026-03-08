//! Cost functions for cutting path optimization.

use crate::config::CuttingConfig;

/// Computes the cost of a cutting sequence.
///
/// Cost = total_rapid_distance + pierce_weight x pierce_count
///
/// This is the primary objective function for sequence optimization.
pub fn sequence_cost(
    total_rapid_distance: f64,
    pierce_count: usize,
    config: &CuttingConfig,
) -> f64 {
    total_rapid_distance + config.pierce_weight * pierce_count as f64
}

/// Computes the Euclidean distance between two points.
#[inline]
pub fn point_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

/// Computes the squared Euclidean distance between two points.
/// Use this when comparing distances (avoids sqrt).
#[inline]
pub fn point_distance_sq(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    dx * dx + dy * dy
}

/// Finds the closest point on a polygon boundary to a given point.
///
/// Returns (closest_point, vertex_index, parameter_t) where:
/// - closest_point is the actual closest point on the boundary
/// - vertex_index is the index of the edge start vertex
/// - parameter_t is the parameter along the edge [0, 1]
pub fn closest_point_on_polygon(
    polygon: &[(f64, f64)],
    point: (f64, f64),
) -> Option<((f64, f64), usize, f64)> {
    if polygon.is_empty() {
        return None;
    }

    let n = polygon.len();
    let mut best_dist_sq = f64::MAX;
    let mut best_point = polygon[0];
    let mut best_idx = 0;
    let mut best_t = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let (px, py) = polygon[i];
        let (qx, qy) = polygon[j];

        let dx = qx - px;
        let dy = qy - py;
        let len_sq = dx * dx + dy * dy;

        let t = if len_sq < 1e-12 {
            0.0
        } else {
            ((point.0 - px) * dx + (point.1 - py) * dy) / len_sq
        };

        let t_clamped = t.clamp(0.0, 1.0);
        let cx = px + t_clamped * dx;
        let cy = py + t_clamped * dy;
        let dist_sq = point_distance_sq(point, (cx, cy));

        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_point = (cx, cy);
            best_idx = i;
            best_t = t_clamped;
        }
    }

    Some((best_point, best_idx, best_t))
}

/// A (start_point, end_point) pair for a contour in the cutting sequence.
pub type ContourEndpoints = ((f64, f64), (f64, f64));

/// Computes the total rapid (non-cutting) distance for a given sequence.
///
/// `positions` contains the (start_point, end_point) for each contour in sequence order.
/// The first rapid move is from `home` to the first contour's start point.
pub fn total_rapid_distance(home: (f64, f64), positions: &[ContourEndpoints]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }

    let mut total = point_distance(home, positions[0].0);

    for i in 1..positions.len() {
        total += point_distance(positions[i - 1].1, positions[i].0);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        assert!((point_distance((0.0, 0.0), (3.0, 4.0)) - 5.0).abs() < 1e-10);
        assert!((point_distance((1.0, 1.0), (1.0, 1.0)) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_distance_sq() {
        assert!((point_distance_sq((0.0, 0.0), (3.0, 4.0)) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_sequence_cost() {
        let config = CuttingConfig::default();
        let cost = sequence_cost(100.0, 5, &config);
        assert!((cost - 150.0).abs() < 1e-10); // 100 + 10*5
    }

    #[test]
    fn test_closest_point_on_polygon_vertex() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (point, _idx, _t) = closest_point_on_polygon(&square, (0.0, 0.0)).unwrap();
        assert!((point.0 - 0.0).abs() < 1e-10);
        assert!((point.1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_closest_point_on_polygon_edge() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (point, idx, _t) = closest_point_on_polygon(&square, (5.0, -1.0)).unwrap();
        assert!((point.0 - 5.0).abs() < 1e-10);
        assert!((point.1 - 0.0).abs() < 1e-10);
        assert_eq!(idx, 0); // Bottom edge
    }

    #[test]
    fn test_closest_point_empty_polygon() {
        assert!(closest_point_on_polygon(&[], (0.0, 0.0)).is_none());
    }

    #[test]
    fn test_total_rapid_distance() {
        let home = (0.0, 0.0);
        let positions = vec![
            ((10.0, 0.0), (10.0, 10.0)), // First contour: pierce at (10,0), end at (10,10)
            ((20.0, 0.0), (20.0, 10.0)), // Second: pierce at (20,0), end at (20,10)
        ];
        let dist = total_rapid_distance(home, &positions);
        // Home to first: 10.0, first end to second start: sqrt((10)^2 + (10)^2)
        let expected = 10.0 + ((10.0_f64).powi(2) + (10.0_f64).powi(2)).sqrt();
        assert!((dist - expected).abs() < 1e-10);
    }
}
