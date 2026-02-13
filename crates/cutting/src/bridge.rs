//! Bridge/tab (micro-joint) placement for cut contours.
//!
//! Places small uncut segments (bridges/tabs) along contour edges to prevent
//! parts from falling through the machine bed during cutting. Bridges enable
//! chain cutting (continuous path across multiple contours) which reduces
//! pierce count.
//!
//! # Placement Rules
//!
//! - Bridges are placed on straight segments only (corner avoidance)
//! - Minimum clearance from vertices (corners) is enforced
//! - Bridges are distributed evenly around the contour perimeter
//! - Number of bridges is determined by contour perimeter and config
//!
//! # References
//!
//! - Hu et al. (2022), "A robust fast bridging algorithm for laser cutting"

use crate::contour::CutContour;
use crate::cost::point_distance;

/// Configuration for bridge/tab placement.
#[derive(Debug, Clone, Copy)]
pub struct BridgeConfig {
    /// Width of each bridge (uncut segment length) in mm.
    pub bridge_width: f64,
    /// Maximum spacing between bridges along the perimeter.
    /// More bridges are added if the spacing exceeds this value.
    pub max_spacing: f64,
    /// Minimum number of bridges per contour.
    pub min_bridges: usize,
    /// Minimum clearance from vertices (corners) in mm.
    /// Bridges won't be placed within this distance of a vertex.
    pub corner_clearance: f64,
    /// Whether bridge placement is enabled.
    pub enabled: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            bridge_width: 2.0,
            max_spacing: 50.0,
            min_bridges: 2,
            corner_clearance: 5.0,
            enabled: false,
        }
    }
}

/// A bridge (micro-joint) placed on a contour edge.
#[derive(Debug, Clone)]
pub struct Bridge {
    /// Start point of the bridge (uncut segment begins here).
    pub start: (f64, f64),
    /// End point of the bridge (uncut segment ends here).
    pub end: (f64, f64),
    /// Midpoint of the bridge.
    pub midpoint: (f64, f64),
    /// Edge index on which the bridge is placed.
    pub edge_index: usize,
    /// Parameter along the edge where the bridge center is [0, 1].
    pub edge_t: f64,
    /// Distance along the perimeter to the bridge center.
    pub perimeter_distance: f64,
}

/// Result of bridge placement for a contour.
#[derive(Debug, Clone)]
pub struct BridgePlacement {
    /// Contour ID.
    pub contour_id: usize,
    /// Placed bridges.
    pub bridges: Vec<Bridge>,
}

/// Places bridges on a contour according to the configuration.
///
/// Bridges are distributed evenly along the perimeter, avoiding vertices
/// (corners) by the configured clearance distance.
pub fn place_bridges(contour: &CutContour, config: &BridgeConfig) -> BridgePlacement {
    if !config.enabled {
        return BridgePlacement {
            contour_id: contour.id,
            bridges: Vec::new(),
        };
    }

    let vertices = &contour.vertices;
    let nv = vertices.len();
    if nv < 3 {
        return BridgePlacement {
            contour_id: contour.id,
            bridges: Vec::new(),
        };
    }

    // Compute edge lengths and cumulative distances
    let mut edge_lengths = Vec::with_capacity(nv);
    let mut cumulative = Vec::with_capacity(nv + 1);
    cumulative.push(0.0);

    for i in 0..nv {
        let j = (i + 1) % nv;
        let len = point_distance(vertices[i], vertices[j]);
        edge_lengths.push(len);
        cumulative.push(cumulative[i] + len);
    }

    let perimeter = cumulative[nv];
    if perimeter < config.bridge_width * 2.0 {
        return BridgePlacement {
            contour_id: contour.id,
            bridges: Vec::new(),
        };
    }

    // Determine number of bridges
    let n_bridges = compute_bridge_count(perimeter, config);

    // Generate candidate positions (equidistant along perimeter)
    let spacing = perimeter / n_bridges as f64;
    let half_width = config.bridge_width / 2.0;

    let mut bridges = Vec::with_capacity(n_bridges);

    for k in 0..n_bridges {
        let target_dist = k as f64 * spacing + spacing / 2.0; // Offset by half spacing
        let target_dist = target_dist % perimeter;

        // Check corner clearance
        if !is_clear_of_corners(target_dist, half_width, &cumulative, config.corner_clearance) {
            // Try to shift the bridge position to find a valid spot
            if let Some(adjusted) = find_valid_position(
                target_dist,
                half_width,
                &cumulative,
                config.corner_clearance,
                perimeter,
            ) {
                if let Some(bridge) = create_bridge(
                    adjusted, half_width, vertices, &cumulative, &edge_lengths, perimeter,
                ) {
                    bridges.push(bridge);
                }
            }
            continue;
        }

        if let Some(bridge) = create_bridge(
            target_dist, half_width, vertices, &cumulative, &edge_lengths, perimeter,
        ) {
            bridges.push(bridge);
        }
    }

    BridgePlacement {
        contour_id: contour.id,
        bridges,
    }
}

/// Computes the number of bridges needed for a contour.
fn compute_bridge_count(perimeter: f64, config: &BridgeConfig) -> usize {
    let count_from_spacing = (perimeter / config.max_spacing).ceil() as usize;
    count_from_spacing.max(config.min_bridges)
}

/// Checks if a bridge position is clear of all corners (vertices).
fn is_clear_of_corners(
    center_dist: f64,
    half_width: f64,
    cumulative: &[f64],
    clearance: f64,
) -> bool {
    let bridge_start = center_dist - half_width;
    let bridge_end = center_dist + half_width;

    // Check distance from each vertex (cumulative distance)
    for &vertex_dist in cumulative.iter() {
        let dist_to_start = (vertex_dist - bridge_start).abs();
        let dist_to_end = (vertex_dist - bridge_end).abs();
        let dist_to_center = (vertex_dist - center_dist).abs();

        if dist_to_center < clearance + half_width
            && (dist_to_start < clearance || dist_to_end < clearance)
        {
            return false;
        }
    }

    true
}

/// Tries to find a valid bridge position near the target.
fn find_valid_position(
    target: f64,
    half_width: f64,
    cumulative: &[f64],
    clearance: f64,
    perimeter: f64,
) -> Option<f64> {
    // Try shifting in small increments
    let step = clearance / 2.0;
    for i in 1..=10 {
        let offset = i as f64 * step;

        let pos_plus = (target + offset) % perimeter;
        if is_clear_of_corners(pos_plus, half_width, cumulative, clearance) {
            return Some(pos_plus);
        }

        let pos_minus = (target - offset + perimeter) % perimeter;
        if is_clear_of_corners(pos_minus, half_width, cumulative, clearance) {
            return Some(pos_minus);
        }
    }

    None
}

/// Creates a Bridge at the given perimeter distance.
fn create_bridge(
    center_dist: f64,
    half_width: f64,
    vertices: &[(f64, f64)],
    cumulative: &[f64],
    edge_lengths: &[f64],
    perimeter: f64,
) -> Option<Bridge> {
    let start_dist = (center_dist - half_width + perimeter) % perimeter;
    let end_dist = (center_dist + half_width) % perimeter;

    let start = point_at_perimeter_distance(start_dist, vertices, cumulative, edge_lengths)?;
    let end = point_at_perimeter_distance(end_dist, vertices, cumulative, edge_lengths)?;
    let (mid, edge_index) =
        point_at_perimeter_distance_with_edge(center_dist, vertices, cumulative, edge_lengths)?;

    let edge_len = edge_lengths[edge_index];
    let edge_t = if edge_len > 1e-12 {
        (center_dist - cumulative[edge_index]) / edge_len
    } else {
        0.0
    };

    Some(Bridge {
        start,
        end,
        midpoint: mid,
        edge_index,
        edge_t,
        perimeter_distance: center_dist,
    })
}

/// Returns the point at a given distance along the contour perimeter.
fn point_at_perimeter_distance(
    dist: f64,
    vertices: &[(f64, f64)],
    cumulative: &[f64],
    edge_lengths: &[f64],
) -> Option<(f64, f64)> {
    point_at_perimeter_distance_with_edge(dist, vertices, cumulative, edge_lengths)
        .map(|(pt, _)| pt)
}

/// Returns the point and edge index at a given perimeter distance.
fn point_at_perimeter_distance_with_edge(
    dist: f64,
    vertices: &[(f64, f64)],
    cumulative: &[f64],
    edge_lengths: &[f64],
) -> Option<((f64, f64), usize)> {
    let nv = vertices.len();
    let perimeter = cumulative[nv];
    if perimeter < 1e-12 {
        return None;
    }

    let d = dist % perimeter;

    for i in 0..nv {
        if d >= cumulative[i] && d <= cumulative[i + 1] + 1e-12 {
            let edge_len = edge_lengths[i];
            if edge_len < 1e-12 {
                return Some((vertices[i], i));
            }
            let t = (d - cumulative[i]) / edge_len;
            let t = t.clamp(0.0, 1.0);
            let j = (i + 1) % nv;
            let px = vertices[i].0 + t * (vertices[j].0 - vertices[i].0);
            let py = vertices[i].1 + t * (vertices[j].1 - vertices[i].1);
            return Some(((px, py), i));
        }
    }

    Some((vertices[nv - 1], nv - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contour::ContourType;

    fn make_rect(id: usize, w: f64, h: f64) -> CutContour {
        CutContour {
            id,
            geometry_id: format!("part{}", id),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
            perimeter: 2.0 * (w + h),
            centroid: (w / 2.0, h / 2.0),
        }
    }

    #[test]
    fn test_disabled_no_bridges() {
        let contour = make_rect(0, 100.0, 50.0);
        let config = BridgeConfig::default(); // enabled = false
        let result = place_bridges(&contour, &config);
        assert!(result.bridges.is_empty());
    }

    #[test]
    fn test_enabled_places_bridges() {
        let contour = make_rect(0, 100.0, 50.0);
        let config = BridgeConfig {
            enabled: true,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);
        assert!(
            result.bridges.len() >= config.min_bridges,
            "Should place at least {} bridges, got {}",
            config.min_bridges,
            result.bridges.len()
        );
    }

    #[test]
    fn test_bridge_count_from_spacing() {
        let contour = make_rect(0, 100.0, 100.0); // perimeter = 400
        let config = BridgeConfig {
            enabled: true,
            max_spacing: 50.0,
            min_bridges: 2,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);
        // perimeter 400 / max_spacing 50 = 8 bridges needed
        assert!(
            result.bridges.len() >= 8,
            "Expected >= 8 bridges for perimeter 400 with spacing 50, got {}",
            result.bridges.len()
        );
    }

    #[test]
    fn test_bridge_count_min() {
        let contour = make_rect(0, 20.0, 20.0); // perimeter = 80
        let config = BridgeConfig {
            enabled: true,
            max_spacing: 100.0, // Would give 1 bridge
            min_bridges: 3,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);
        assert!(
            result.bridges.len() >= 3,
            "Should respect min_bridges=3, got {}",
            result.bridges.len()
        );
    }

    #[test]
    fn test_bridge_midpoint_on_edge() {
        let contour = make_rect(0, 100.0, 50.0);
        let config = BridgeConfig {
            enabled: true,
            corner_clearance: 3.0,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);

        for bridge in &result.bridges {
            // Midpoint should be on the contour boundary
            let (mx, my) = bridge.midpoint;
            let on_boundary = (-0.1..=100.1).contains(&mx) && (-0.1..=50.1).contains(&my);
            assert!(on_boundary, "Bridge midpoint ({}, {}) should be on contour boundary", mx, my);
        }
    }

    #[test]
    fn test_bridge_width() {
        let contour = make_rect(0, 100.0, 50.0);
        let config = BridgeConfig {
            enabled: true,
            bridge_width: 3.0,
            corner_clearance: 3.0,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);

        for bridge in &result.bridges {
            let width = point_distance(bridge.start, bridge.end);
            // Width should be approximately bridge_width (may differ slightly at corners)
            assert!(
                width < config.bridge_width + 1.0,
                "Bridge width {} should be close to {}",
                width,
                config.bridge_width
            );
        }
    }

    #[test]
    fn test_small_contour_no_bridges() {
        // Contour too small for bridges
        let contour = CutContour {
            id: 0,
            geometry_id: "tiny".to_string(),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)],
            perimeter: 3.0,
            centroid: (0.5, 0.17),
        };
        let config = BridgeConfig {
            enabled: true,
            bridge_width: 2.0,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);
        assert!(result.bridges.is_empty(), "Too-small contour should get no bridges");
    }

    #[test]
    fn test_default_config() {
        let config = BridgeConfig::default();
        assert!(!config.enabled);
        assert!((config.bridge_width - 2.0).abs() < 1e-10);
        assert!((config.max_spacing - 50.0).abs() < 1e-10);
        assert_eq!(config.min_bridges, 2);
        assert!((config.corner_clearance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bridge_edge_index_valid() {
        let contour = make_rect(0, 100.0, 50.0);
        let config = BridgeConfig {
            enabled: true,
            ..BridgeConfig::default()
        };
        let result = place_bridges(&contour, &config);

        for bridge in &result.bridges {
            assert!(
                bridge.edge_index < contour.vertices.len(),
                "Edge index {} should be < vertex count {}",
                bridge.edge_index,
                contour.vertices.len()
            );
            assert!(bridge.edge_t >= 0.0 && bridge.edge_t <= 1.0);
        }
    }
}
