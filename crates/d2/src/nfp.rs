//! No-Fit Polygon (NFP) computation.
//!
//! The NFP of two polygons A and B represents all positions where the reference
//! point of B can be placed such that B touches or overlaps A.
//!
//! This module implements:
//! - **Convex case**: Minkowski sum algorithm (O(n+m) for convex polygons)
//! - **Non-convex case**: Convex decomposition + union approach using `i_overlay`

use crate::geometry::Geometry2D;
use geo::{ConvexHull, Coord, LineString};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};
use u_nesting_core::geometry::Geometry2DExt;
use u_nesting_core::{Error, Result};

/// NFP computation result.
#[derive(Debug, Clone)]
pub struct Nfp {
    /// The computed NFP polygon(s).
    /// Multiple polygons can result from non-convex shapes.
    pub polygons: Vec<Vec<(f64, f64)>>,
}

impl Nfp {
    /// Creates a new empty NFP.
    pub fn new() -> Self {
        Self {
            polygons: Vec::new(),
        }
    }

    /// Creates an NFP with a single polygon.
    pub fn from_polygon(polygon: Vec<(f64, f64)>) -> Self {
        Self {
            polygons: vec![polygon],
        }
    }

    /// Creates an NFP with multiple polygons.
    pub fn from_polygons(polygons: Vec<Vec<(f64, f64)>>) -> Self {
        Self { polygons }
    }

    /// Returns true if the NFP is empty.
    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty()
    }

    /// Returns the total vertex count across all polygons.
    pub fn vertex_count(&self) -> usize {
        self.polygons.iter().map(|p| p.len()).sum()
    }
}

impl Default for Nfp {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the No-Fit Polygon between two geometries.
///
/// The NFP represents all positions where the orbiting polygon would
/// overlap with the stationary polygon.
///
/// # Algorithm Selection
/// - If both polygons are convex: uses fast Minkowski sum (O(n+m))
/// - Otherwise: uses convex decomposition + union approach
///
/// # Arguments
/// * `stationary` - The fixed polygon
/// * `orbiting` - The polygon to be placed
/// * `rotation` - Rotation angle of the orbiting polygon in radians
///
/// # Returns
/// The computed NFP, or an error if computation fails.
pub fn compute_nfp(
    stationary: &Geometry2D,
    orbiting: &Geometry2D,
    rotation: f64,
) -> Result<Nfp> {
    // Get the polygons
    let stat_exterior = stationary.exterior();
    let orb_exterior = orbiting.exterior();

    if stat_exterior.len() < 3 || orb_exterior.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Polygons must have at least 3 vertices".into(),
        ));
    }

    // Apply rotation to orbiting polygon
    let rotated_orbiting = rotate_polygon(orb_exterior, rotation);

    // Check if both are convex for fast path
    if stationary.is_convex()
        && is_polygon_convex(&rotated_orbiting)
        && stationary.holes().is_empty()
    {
        // Fast path: Minkowski sum for convex polygons
        compute_nfp_convex(stat_exterior, &rotated_orbiting)
    } else {
        // General case: decomposition + union
        compute_nfp_general(stationary, &rotated_orbiting)
    }
}

/// Computes the Inner-Fit Polygon (IFP) of a geometry within a boundary.
///
/// The IFP represents all valid positions where the reference point of
/// a geometry can be placed within the boundary.
///
/// # Arguments
/// * `boundary_polygon` - The boundary polygon vertices (counter-clockwise)
/// * `geometry` - The geometry to fit inside
/// * `rotation` - Rotation angle of the geometry in radians
///
/// # Returns
/// The computed IFP, or an error if computation fails.
pub fn compute_ifp(
    boundary_polygon: &[(f64, f64)],
    geometry: &Geometry2D,
    rotation: f64,
) -> Result<Nfp> {
    compute_ifp_with_margin(boundary_polygon, geometry, rotation, 0.0)
}

/// Computes the Inner-Fit Polygon (IFP) of a geometry within a boundary with margin.
///
/// The IFP represents all valid positions where the reference point of
/// a geometry can be placed within the boundary, accounting for a margin
/// (offset) from the boundary edges.
///
/// # Arguments
/// * `boundary_polygon` - The boundary polygon vertices (counter-clockwise)
/// * `geometry` - The geometry to fit inside
/// * `rotation` - Rotation angle of the geometry in radians
/// * `margin` - Distance to maintain from boundary edges (applied to both boundary and geometry)
///
/// # Returns
/// The computed IFP, or an error if computation fails.
pub fn compute_ifp_with_margin(
    boundary_polygon: &[(f64, f64)],
    geometry: &Geometry2D,
    rotation: f64,
    margin: f64,
) -> Result<Nfp> {
    if boundary_polygon.len() < 3 {
        return Err(Error::InvalidBoundary(
            "Boundary must have at least 3 vertices".into(),
        ));
    }

    let geom_exterior = geometry.exterior();
    if geom_exterior.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Geometry must have at least 3 vertices".into(),
        ));
    }

    // Apply rotation to geometry
    let rotated_geom = rotate_polygon(geom_exterior, rotation);

    // Apply margin by shrinking the boundary inward
    let effective_boundary = if margin > 0.0 {
        shrink_polygon(boundary_polygon, margin)?
    } else {
        boundary_polygon.to_vec()
    };

    if effective_boundary.len() < 3 {
        return Err(Error::InvalidBoundary(
            "Boundary too small after applying margin".into(),
        ));
    }

    // The IFP is computed by:
    // 1. Reflect the geometry about its reference point (negate all coordinates)
    // 2. Compute Minkowski sum of boundary and reflected geometry
    // This gives us the locus of valid placement positions

    let reflected_geom: Vec<(f64, f64)> = rotated_geom.iter().map(|&(x, y)| (-x, -y)).collect();

    // Check if both are convex for fast path
    if is_polygon_convex(&effective_boundary) && is_polygon_convex(&reflected_geom) {
        compute_minkowski_sum_convex(&effective_boundary, &reflected_geom)
    } else {
        // For non-convex cases, use general approach
        compute_minkowski_sum_general(&effective_boundary, &reflected_geom)
    }
}

/// Shrinks a polygon by moving all edges inward by the given offset.
///
/// For axis-aligned rectangles (the common case for boundaries), this shrinks
/// each edge inward. For general polygons, it uses a vertex-based approach.
fn shrink_polygon(polygon: &[(f64, f64)], offset: f64) -> Result<Vec<(f64, f64)>> {
    if polygon.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Polygon must have at least 3 vertices".into(),
        ));
    }

    // Check if this is an axis-aligned rectangle (common case for boundaries)
    if polygon.len() == 4 {
        let (min_x, min_y, max_x, max_y) = bounding_box(polygon);

        // Check if all vertices are on the bounding box edges (axis-aligned)
        let is_axis_aligned = polygon.iter().all(|&(x, y)| {
            ((x - min_x).abs() < 1e-10 || (x - max_x).abs() < 1e-10)
                && ((y - min_y).abs() < 1e-10 || (y - max_y).abs() < 1e-10)
        });

        if is_axis_aligned {
            // Simple shrink for axis-aligned rectangle
            let new_min_x = min_x + offset;
            let new_min_y = min_y + offset;
            let new_max_x = max_x - offset;
            let new_max_y = max_y - offset;

            // Check if still valid
            if new_min_x >= new_max_x || new_min_y >= new_max_y {
                return Err(Error::InvalidGeometry(
                    "Offset polygon collapsed".into(),
                ));
            }

            return Ok(vec![
                (new_min_x, new_min_y),
                (new_max_x, new_min_y),
                (new_max_x, new_max_y),
                (new_min_x, new_max_y),
            ]);
        }
    }

    // General polygon shrink using centroid-based approach
    let (cx, cy) = polygon_centroid(polygon);

    let result: Vec<(f64, f64)> = polygon
        .iter()
        .filter_map(|&(x, y)| {
            let dx = x - cx;
            let dy = y - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < offset + 1e-10 {
                // Vertex too close to centroid
                return None;
            }

            // Move vertex toward centroid by offset
            let factor = (dist - offset) / dist;
            Some((cx + dx * factor, cy + dy * factor))
        })
        .collect();

    // Validate result polygon has reasonable size
    if result.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Offset polygon collapsed".into(),
        ));
    }

    // Check if the polygon has positive area (not self-intersecting)
    let area = signed_area(&result).abs();
    if area <= 1e-10 {
        return Err(Error::InvalidGeometry(
            "Offset polygon collapsed".into(),
        ));
    }

    Ok(result)
}

/// Computes bounding box of a polygon.
fn bounding_box(polygon: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &(x, y) in polygon {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

/// Computes polygon centroid.
fn polygon_centroid(polygon: &[(f64, f64)]) -> (f64, f64) {
    let n = polygon.len() as f64;
    let sum_x: f64 = polygon.iter().map(|p| p.0).sum();
    let sum_y: f64 = polygon.iter().map(|p| p.1).sum();
    (sum_x / n, sum_y / n)
}

/// Computes NFP for two convex polygons using Minkowski sum.
///
/// For the NFP, we compute: A ⊕ (-B) where ⊕ is Minkowski sum
/// This gives all positions where B's reference point would cause overlap with A.
fn compute_nfp_convex(stationary: &[(f64, f64)], orbiting: &[(f64, f64)]) -> Result<Nfp> {
    // Reflect the orbiting polygon about origin (negate coordinates)
    let reflected: Vec<(f64, f64)> = orbiting.iter().map(|&(x, y)| (-x, -y)).collect();

    compute_minkowski_sum_convex(stationary, &reflected)
}

/// Computes Minkowski sum of two convex polygons.
///
/// Algorithm: Merge sorted edge vectors from both polygons.
/// Time complexity: O(n + m) where n, m are vertex counts.
fn compute_minkowski_sum_convex(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> Result<Nfp> {
    // Ensure polygons are in counter-clockwise order
    let a = ensure_ccw(poly_a);
    let b = ensure_ccw(poly_b);

    // Get edge vectors for both polygons
    let edges_a = get_edge_vectors(&a);
    let edges_b = get_edge_vectors(&b);

    // Find starting vertices (bottom-most, then left-most)
    let start_a = find_bottom_left_vertex(&a);
    let start_b = find_bottom_left_vertex(&b);

    // Starting point of Minkowski sum
    let start_point = (a[start_a].0 + b[start_b].0, a[start_a].1 + b[start_b].1);

    // Merge edge vectors by angle
    let merged_edges = merge_edge_vectors(&edges_a, start_a, &edges_b, start_b);

    // Build the result polygon by following merged edges
    let mut result = Vec::with_capacity(merged_edges.len() + 1);
    let mut current = start_point;
    result.push(current);

    for (dx, dy) in merged_edges.iter() {
        current = (current.0 + dx, current.1 + dy);
        result.push(current);
    }

    // Remove the last point if it's the same as the first (closed polygon)
    if result.len() > 1 {
        let first = result[0];
        let last = result[result.len() - 1];
        if (first.0 - last.0).abs() < 1e-10 && (first.1 - last.1).abs() < 1e-10 {
            result.pop();
        }
    }

    Ok(Nfp::from_polygon(result))
}

/// Computes NFP for non-convex polygons using convex decomposition + union.
///
/// The algorithm:
/// 1. Decompose both polygons into convex parts (using triangulation)
/// 2. Compute pairwise Minkowski sums of convex parts
/// 3. Union all partial results using `i_overlay`
fn compute_nfp_general(stationary: &Geometry2D, rotated_orbiting: &[(f64, f64)]) -> Result<Nfp> {
    // Triangulate both polygons into convex parts
    let stat_triangles = triangulate_polygon(stationary.exterior());
    let orb_triangles = triangulate_polygon(rotated_orbiting);

    if stat_triangles.is_empty() || orb_triangles.is_empty() {
        // Fall back to convex hull approximation
        let stat_hull = stationary.convex_hull();
        let orb_hull = convex_hull_of_points(rotated_orbiting);
        let reflected: Vec<(f64, f64)> = orb_hull.iter().map(|&(x, y)| (-x, -y)).collect();
        return compute_minkowski_sum_convex(&stat_hull, &reflected);
    }

    // Compute pairwise Minkowski sums
    let mut partial_nfps: Vec<Vec<(f64, f64)>> = Vec::new();

    for stat_tri in &stat_triangles {
        for orb_tri in &orb_triangles {
            // Reflect orbiting triangle
            let reflected: Vec<(f64, f64)> = orb_tri.iter().map(|&(x, y)| (-x, -y)).collect();

            // Compute Minkowski sum of two convex polygons
            if let Ok(nfp) = compute_minkowski_sum_convex(stat_tri, &reflected) {
                for polygon in nfp.polygons {
                    if polygon.len() >= 3 {
                        partial_nfps.push(polygon);
                    }
                }
            }
        }
    }

    if partial_nfps.is_empty() {
        // Fall back to convex hull
        let stat_hull = stationary.convex_hull();
        let orb_hull = convex_hull_of_points(rotated_orbiting);
        let reflected: Vec<(f64, f64)> = orb_hull.iter().map(|&(x, y)| (-x, -y)).collect();
        return compute_minkowski_sum_convex(&stat_hull, &reflected);
    }

    // Union all partial NFPs using i_overlay
    union_polygons(&partial_nfps)
}

/// Triangulates a polygon into convex parts (ear clipping algorithm).
fn triangulate_polygon(polygon: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    if polygon.len() < 3 {
        return Vec::new();
    }

    // For convex polygons, just return the polygon itself
    if is_polygon_convex(polygon) {
        return vec![polygon.to_vec()];
    }

    // Simple ear-clipping triangulation
    let mut vertices: Vec<(f64, f64)> = ensure_ccw(polygon);
    let mut triangles = Vec::new();

    while vertices.len() > 3 {
        let n = vertices.len();
        let mut ear_found = false;

        for i in 0..n {
            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;

            // Check if this is an ear (convex vertex with no other vertices inside)
            if is_ear(&vertices, prev, i, next) {
                triangles.push(vec![vertices[prev], vertices[i], vertices[next]]);
                vertices.remove(i);
                ear_found = true;
                break;
            }
        }

        if !ear_found {
            // No ear found, polygon might be degenerate
            // Fall back to returning the convex hull
            return vec![convex_hull_of_points(polygon)];
        }
    }

    if vertices.len() == 3 {
        triangles.push(vertices);
    }

    triangles
}

/// Checks if vertex i forms an ear in the polygon.
fn is_ear(vertices: &[(f64, f64)], prev: usize, curr: usize, next: usize) -> bool {
    let (ax, ay) = vertices[prev];
    let (bx, by) = vertices[curr];
    let (cx, cy) = vertices[next];

    // Check if the vertex is convex (turn left in CCW polygon)
    let cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if cross <= 0.0 {
        return false; // Reflex vertex, not an ear
    }

    // Check if any other vertex is inside this triangle
    for (i, &(px, py)) in vertices.iter().enumerate() {
        if i == prev || i == curr || i == next {
            continue;
        }
        if point_in_triangle((px, py), (ax, ay), (bx, by), (cx, cy)) {
            return false;
        }
    }

    true
}

/// Checks if a point is inside a triangle.
fn point_in_triangle(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let (px, py) = p;
    let (ax, ay) = a;
    let (bx, by) = b;
    let (cx, cy) = c;

    let v0 = (cx - ax, cy - ay);
    let v1 = (bx - ax, by - ay);
    let v2 = (px - ax, py - ay);

    let dot00 = v0.0 * v0.0 + v0.1 * v0.1;
    let dot01 = v0.0 * v1.0 + v0.1 * v1.1;
    let dot02 = v0.0 * v2.0 + v0.1 * v2.1;
    let dot11 = v1.0 * v1.0 + v1.1 * v1.1;
    let dot12 = v1.0 * v2.0 + v1.1 * v2.1;

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u > 1e-10 && v > 1e-10 && (u + v) < 1.0 - 1e-10
}

/// Unions multiple polygons using i_overlay.
fn union_polygons(polygons: &[Vec<(f64, f64)>]) -> Result<Nfp> {
    if polygons.is_empty() {
        return Ok(Nfp::new());
    }

    if polygons.len() == 1 {
        return Ok(Nfp::from_polygon(polygons[0].clone()));
    }

    // Start with the first polygon
    let mut result: Vec<Vec<[f64; 2]>> = vec![polygons[0]
        .iter()
        .map(|&(x, y)| [x, y])
        .collect()];

    // Union with each subsequent polygon
    for polygon in &polygons[1..] {
        let clip: Vec<[f64; 2]> = polygon.iter().map(|&(x, y)| [x, y]).collect();

        // Perform union using i_overlay
        let shapes = result.overlay(&[clip], OverlayRule::Union, FillRule::NonZero);

        // Convert shapes back to our format
        result = Vec::new();
        for shape in shapes {
            for contour in shape {
                if contour.len() >= 3 {
                    result.push(contour);
                }
            }
        }

        if result.is_empty() {
            // Union failed, continue with remaining polygons
            continue;
        }
    }

    // Convert back to our Nfp format
    let nfp_polygons: Vec<Vec<(f64, f64)>> = result
        .into_iter()
        .map(|contour| contour.into_iter().map(|[x, y]| (x, y)).collect())
        .collect();

    if nfp_polygons.is_empty() {
        // Fall back to returning the first polygon
        return Ok(Nfp::from_polygon(polygons[0].clone()));
    }

    Ok(Nfp::from_polygons(nfp_polygons))
}

/// Computes Minkowski sum for general (non-convex) polygons.
fn compute_minkowski_sum_general(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> Result<Nfp> {
    // Use convex hull approximation for general case
    let hull_a = convex_hull_of_points(poly_a);
    let hull_b = convex_hull_of_points(poly_b);

    compute_minkowski_sum_convex(&hull_a, &hull_b)
}

// ============================================================================
// Helper functions
// ============================================================================

/// Rotates a polygon around the origin by the given angle (in radians).
fn rotate_polygon(polygon: &[(f64, f64)], angle: f64) -> Vec<(f64, f64)> {
    if angle.abs() < 1e-10 {
        return polygon.to_vec();
    }

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    polygon
        .iter()
        .map(|&(x, y)| (x * cos_a - y * sin_a, x * sin_a + y * cos_a))
        .collect()
}

/// Checks if a polygon is convex.
fn is_polygon_convex(polygon: &[(f64, f64)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let n = polygon.len();
    let mut sign = 0i32;

    for i in 0..n {
        let p0 = polygon[i];
        let p1 = polygon[(i + 1) % n];
        let p2 = polygon[(i + 2) % n];

        let cross = (p1.0 - p0.0) * (p2.1 - p1.1) - (p1.1 - p0.1) * (p2.0 - p1.0);

        if cross.abs() > 1e-10 {
            let current_sign = if cross > 0.0 { 1 } else { -1 };
            if sign == 0 {
                sign = current_sign;
            } else if sign != current_sign {
                return false;
            }
        }
    }

    true
}

/// Ensures polygon vertices are in counter-clockwise order.
fn ensure_ccw(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if signed_area(polygon) < 0.0 {
        polygon.iter().rev().cloned().collect()
    } else {
        polygon.to_vec()
    }
}

/// Computes the signed area of a polygon.
/// Positive for counter-clockwise, negative for clockwise.
fn signed_area(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    let mut area = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        area += polygon[i].0 * polygon[j].1;
        area -= polygon[j].0 * polygon[i].1;
    }

    area / 2.0
}

/// Gets edge vectors from a polygon.
fn get_edge_vectors(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = polygon.len();
    (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            (polygon[j].0 - polygon[i].0, polygon[j].1 - polygon[i].1)
        })
        .collect()
}

/// Finds the index of the bottom-most (then left-most) vertex.
fn find_bottom_left_vertex(polygon: &[(f64, f64)]) -> usize {
    let mut min_idx = 0;

    for (i, &(x, y)) in polygon.iter().enumerate() {
        let (min_x, min_y) = polygon[min_idx];
        if y < min_y || (y == min_y && x < min_x) {
            min_idx = i;
        }
    }

    min_idx
}

/// Computes the angle of an edge vector (in radians, 0 to 2π).
fn edge_angle(dx: f64, dy: f64) -> f64 {
    let angle = dy.atan2(dx);
    if angle < 0.0 {
        angle + 2.0 * PI
    } else {
        angle
    }
}

/// Merges edge vectors from two polygons by angle for Minkowski sum.
fn merge_edge_vectors(
    edges_a: &[(f64, f64)],
    start_a: usize,
    edges_b: &[(f64, f64)],
    start_b: usize,
) -> Vec<(f64, f64)> {
    let n_a = edges_a.len();
    let n_b = edges_b.len();
    let total = n_a + n_b;

    let mut result = Vec::with_capacity(total);
    let mut i_a = 0;
    let mut i_b = 0;

    while i_a < n_a || i_b < n_b {
        if i_a >= n_a {
            // Only edges from B remaining
            let idx = (start_b + i_b) % n_b;
            result.push(edges_b[idx]);
            i_b += 1;
        } else if i_b >= n_b {
            // Only edges from A remaining
            let idx = (start_a + i_a) % n_a;
            result.push(edges_a[idx]);
            i_a += 1;
        } else {
            // Compare angles
            let idx_a = (start_a + i_a) % n_a;
            let idx_b = (start_b + i_b) % n_b;

            let angle_a = edge_angle(edges_a[idx_a].0, edges_a[idx_a].1);
            let angle_b = edge_angle(edges_b[idx_b].0, edges_b[idx_b].1);

            if angle_a <= angle_b + 1e-10 {
                result.push(edges_a[idx_a]);
                i_a += 1;
            }
            if angle_b <= angle_a + 1e-10 {
                result.push(edges_b[idx_b]);
                i_b += 1;
            }
        }
    }

    result
}

/// Computes convex hull of a set of points.
fn convex_hull_of_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let coords: Vec<Coord<f64>> = points.iter().map(|&(x, y)| Coord { x, y }).collect();

    let line_string = LineString::from(coords);
    let hull = line_string.convex_hull();

    hull.exterior()
        .coords()
        .map(|c| (c.x, c.y))
        .collect::<Vec<_>>()
        .into_iter()
        .take(hull.exterior().coords().count().saturating_sub(1)) // Remove duplicate closing point
        .collect()
}

// ============================================================================
// NFP-guided placement helpers
// ============================================================================

/// Checks if a point is inside a polygon (using ray casting algorithm).
pub fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let (px, py) = point;
    let n = polygon.len();
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Checks if a point is outside all given NFP polygons (not overlapping any placed piece).
pub fn point_outside_all_nfps(point: (f64, f64), nfps: &[&Nfp]) -> bool {
    for nfp in nfps {
        for polygon in &nfp.polygons {
            if point_in_polygon(point, polygon) {
                return false;
            }
        }
    }
    true
}

/// Finds the bottom-left valid placement point.
///
/// The valid region is defined as points that are:
/// 1. Inside the IFP (Inner-Fit Polygon) - the boundary constraint
/// 2. Outside all NFPs (No-Fit Polygons) - not overlapping placed pieces
///
/// # Arguments
/// * `ifp` - The inner-fit polygon (valid positions within boundary)
/// * `nfps` - List of NFPs with already placed pieces
/// * `sample_step` - Grid sampling step size (smaller = more accurate but slower)
///
/// # Returns
/// The bottom-left valid point, or None if no valid position exists.
pub fn find_bottom_left_placement(
    ifp: &Nfp,
    nfps: &[&Nfp],
    sample_step: f64,
) -> Option<(f64, f64)> {
    if ifp.is_empty() {
        return None;
    }

    // First, try the vertices of the IFP (often optimal positions)
    let mut candidates: Vec<(f64, f64)> = Vec::new();

    for polygon in &ifp.polygons {
        candidates.extend(polygon.iter().copied());
    }

    // Also collect NFP vertices as potential optimal positions
    for nfp in nfps {
        for polygon in &nfp.polygons {
            candidates.extend(polygon.iter().copied());
        }
    }

    // Find the bounding box of the IFP for grid sampling
    let (min_x, min_y, max_x, max_y) = ifp_bounding_box(ifp);

    // Add grid sample points
    let mut y = min_y;
    while y <= max_y {
        let mut x = min_x;
        while x <= max_x {
            candidates.push((x, y));
            x += sample_step;
        }
        y += sample_step;
    }

    // Filter candidates to those inside IFP and outside all NFPs
    let valid_candidates: Vec<(f64, f64)> = candidates
        .into_iter()
        .filter(|&point| {
            // Must be inside IFP
            let in_ifp = ifp.polygons.iter().any(|p| point_in_polygon(point, p));
            if !in_ifp {
                return false;
            }
            // Must be outside all NFPs
            point_outside_all_nfps(point, nfps)
        })
        .collect();

    // Find bottom-left point (minimize y first, then x)
    valid_candidates
        .into_iter()
        .min_by(|a, b| {
            // Compare y first (bottom), then x (left)
            match a.1.partial_cmp(&b.1) {
                Some(std::cmp::Ordering::Equal) => {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                }
                Some(ord) => ord,
                None => std::cmp::Ordering::Equal,
            }
        })
}

/// Computes the bounding box of an NFP.
fn ifp_bounding_box(ifp: &Nfp) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for polygon in &ifp.polygons {
        for &(x, y) in polygon {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    (min_x, min_y, max_x, max_y)
}

/// Represents a placed geometry for NFP computation.
#[derive(Debug, Clone)]
pub struct PlacedGeometry {
    /// The original geometry.
    pub geometry: Geometry2D,
    /// The placement position (x, y).
    pub position: (f64, f64),
    /// The rotation angle in radians.
    pub rotation: f64,
}

impl PlacedGeometry {
    /// Creates a new placed geometry.
    pub fn new(geometry: Geometry2D, position: (f64, f64), rotation: f64) -> Self {
        Self {
            geometry,
            position,
            rotation,
        }
    }

    /// Returns the translated polygon vertices.
    pub fn translated_exterior(&self) -> Vec<(f64, f64)> {
        let rotated = rotate_polygon(self.geometry.exterior(), self.rotation);
        rotated
            .into_iter()
            .map(|(x, y)| (x + self.position.0, y + self.position.1))
            .collect()
    }
}

// ============================================================================
// NFP Cache
// ============================================================================

/// Cache key for NFP lookups.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NfpCacheKey {
    geometry_a: String,
    geometry_b: String,
    rotation_millideg: i32, // Rotation in millidegrees for integer key
}

impl NfpCacheKey {
    fn new(id_a: &str, id_b: &str, rotation_rad: f64) -> Self {
        // Convert radians to millidegrees for integer key
        let rotation_millideg = ((rotation_rad * 180.0 / PI) * 1000.0).round() as i32;
        Self {
            geometry_a: id_a.to_string(),
            geometry_b: id_b.to_string(),
            rotation_millideg,
        }
    }
}

/// Thread-safe NFP cache for storing precomputed NFPs.
#[derive(Debug)]
pub struct NfpCache {
    cache: RwLock<HashMap<NfpCacheKey, Arc<Nfp>>>,
    max_size: usize,
}

impl NfpCache {
    /// Creates a new NFP cache with default capacity (1000 entries).
    pub fn new() -> Self {
        Self::with_capacity(1000)
    }

    /// Creates a new NFP cache with specified capacity.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size,
        }
    }

    /// Gets a cached NFP or computes and caches it.
    ///
    /// # Arguments
    /// * `key` - Tuple of (geometry_id_a, geometry_id_b, rotation_in_radians)
    /// * `compute` - Function to compute the NFP if not cached
    pub fn get_or_compute<F>(&self, key: (&str, &str, f64), compute: F) -> Result<Arc<Nfp>>
    where
        F: FnOnce() -> Result<Nfp>,
    {
        let cache_key = NfpCacheKey::new(key.0, key.1, key.2);

        // Try to get from cache first (read lock)
        {
            let cache = self.cache.read().map_err(|e| {
                Error::Internal(format!("Failed to acquire cache read lock: {}", e))
            })?;
            if let Some(nfp) = cache.get(&cache_key) {
                return Ok(Arc::clone(nfp));
            }
        }

        // Compute the NFP
        let nfp = Arc::new(compute()?);

        // Store in cache (write lock)
        {
            let mut cache = self.cache.write().map_err(|e| {
                Error::Internal(format!("Failed to acquire cache write lock: {}", e))
            })?;

            // Simple eviction: if at capacity, clear half the cache
            if cache.len() >= self.max_size {
                let keys_to_remove: Vec<_> =
                    cache.keys().take(self.max_size / 2).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }

            cache.insert(cache_key, Arc::clone(&nfp));
        }

        Ok(nfp)
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
}

impl Default for NfpCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn rect(w: f64, h: f64) -> Vec<(f64, f64)> {
        vec![(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    }

    fn triangle() -> Vec<(f64, f64)> {
        vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    }

    #[test]
    fn test_is_polygon_convex() {
        // Square is convex
        assert!(is_polygon_convex(&rect(10.0, 10.0)));

        // Triangle is convex
        assert!(is_polygon_convex(&triangle()));

        // L-shape is not convex
        let l_shape = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (5.0, 5.0),
            (5.0, 10.0),
            (0.0, 10.0),
        ];
        assert!(!is_polygon_convex(&l_shape));
    }

    #[test]
    fn test_signed_area() {
        // CCW square has positive area
        let ccw_square = rect(10.0, 10.0);
        assert!(signed_area(&ccw_square) > 0.0);
        assert_relative_eq!(signed_area(&ccw_square).abs(), 100.0, epsilon = 1e-10);

        // CW square has negative area
        let cw_square: Vec<_> = ccw_square.into_iter().rev().collect();
        assert!(signed_area(&cw_square) < 0.0);
    }

    #[test]
    fn test_rotate_polygon() {
        let square = rect(10.0, 10.0);

        // No rotation
        let rotated = rotate_polygon(&square, 0.0);
        assert_eq!(rotated.len(), square.len());

        // 90 degree rotation
        let rotated = rotate_polygon(&[(1.0, 0.0)], PI / 2.0);
        assert_relative_eq!(rotated[0].0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[0].1, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nfp_two_squares() {
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        let nfp = compute_nfp(&a, &b, 0.0).unwrap();

        assert!(!nfp.is_empty());
        assert_eq!(nfp.polygons.len(), 1);

        // NFP of two axis-aligned rectangles should have 4 vertices
        // NFP dimensions should be (10+5) x (10+5) = 15 x 15
        let polygon = &nfp.polygons[0];
        assert!(polygon.len() >= 4);
    }

    #[test]
    fn test_nfp_with_rotation() {
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        // Compute with 45 degree rotation
        let nfp = compute_nfp(&a, &b, PI / 4.0).unwrap();

        assert!(!nfp.is_empty());
        // Rotated NFP should have more vertices due to the octagonal shape
    }

    #[test]
    fn test_ifp_square_in_boundary() {
        let boundary = rect(100.0, 100.0);
        let geom = Geometry2D::rectangle("G", 10.0, 10.0);

        let ifp = compute_ifp(&boundary, &geom, 0.0).unwrap();

        assert!(!ifp.is_empty());
        // IFP should be a rectangle of size (100-10) x (100-10) = 90 x 90
    }

    #[test]
    fn test_nfp_cache() {
        let cache = NfpCache::new();

        let compute_count = std::sync::atomic::AtomicUsize::new(0);

        let result1 = cache
            .get_or_compute(("A", "B", 0.0), || {
                compute_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Nfp::from_polygon(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]))
            })
            .unwrap();

        let result2 = cache
            .get_or_compute(("A", "B", 0.0), || {
                compute_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Nfp::from_polygon(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]))
            })
            .unwrap();

        // Should only compute once
        assert_eq!(
            compute_count.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(result1.polygons, result2.polygons);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_nfp_cache_different_rotations() {
        let cache = NfpCache::new();

        cache
            .get_or_compute(("A", "B", 0.0), || {
                Ok(Nfp::from_polygon(vec![(0.0, 0.0), (1.0, 0.0)]))
            })
            .unwrap();

        cache
            .get_or_compute(("A", "B", PI / 2.0), || {
                Ok(Nfp::from_polygon(vec![(0.0, 0.0), (0.0, 1.0)]))
            })
            .unwrap();

        // Different rotations should be cached separately
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_edge_angle() {
        // Right (0 degrees)
        assert_relative_eq!(edge_angle(1.0, 0.0), 0.0, epsilon = 1e-10);

        // Up (90 degrees)
        assert_relative_eq!(edge_angle(0.0, 1.0), PI / 2.0, epsilon = 1e-10);

        // Left (180 degrees)
        assert_relative_eq!(edge_angle(-1.0, 0.0), PI, epsilon = 1e-10);

        // Down (270 degrees)
        assert_relative_eq!(edge_angle(0.0, -1.0), 3.0 * PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_convex_hull_of_points() {
        let points = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (5.0, 5.0), // Interior point
            (10.0, 10.0),
            (0.0, 10.0),
        ];

        let hull = convex_hull_of_points(&points);

        // Hull should have 4 vertices (square without interior point)
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_shrink_polygon_square() {
        let square = rect(100.0, 100.0);
        let shrunk = shrink_polygon(&square, 10.0).unwrap();

        // Should still have 4 vertices
        assert_eq!(shrunk.len(), 4);

        // The shrunk polygon should be smaller
        let original_area = signed_area(&square).abs();
        let shrunk_area = signed_area(&shrunk).abs();
        assert!(
            shrunk_area < original_area,
            "shrunk_area ({}) should be < original_area ({})",
            shrunk_area,
            original_area
        );

        // Expected area: (100-20)*(100-20) = 6400
        // (10.0 offset on each side)
        assert_relative_eq!(shrunk_area, 6400.0, epsilon = 1.0);
    }

    #[test]
    fn test_shrink_polygon_collapse() {
        let small_square = rect(10.0, 10.0);

        // Shrinking by 6 should collapse the 10x10 polygon (becomes 0 or negative)
        let result = shrink_polygon(&small_square, 6.0);
        assert!(result.is_err(), "Polygon should collapse when offset >= width/2");
    }

    #[test]
    fn test_ifp_with_margin() {
        let boundary = rect(100.0, 100.0);
        let geom = Geometry2D::rectangle("G", 10.0, 10.0);

        // Without margin
        let ifp_no_margin = compute_ifp(&boundary, &geom, 0.0).unwrap();

        // With margin
        let ifp_with_margin = compute_ifp_with_margin(&boundary, &geom, 0.0, 5.0).unwrap();

        assert!(!ifp_no_margin.is_empty());
        assert!(!ifp_with_margin.is_empty());

        // IFP with margin should be smaller
        let (min_x_no, _min_y_no, max_x_no, _max_y_no) = ifp_bounding_box(&ifp_no_margin);
        let (min_x_margin, _min_y_margin, max_x_margin, _max_y_margin) =
            ifp_bounding_box(&ifp_with_margin);

        let width_no = max_x_no - min_x_no;
        let width_margin = max_x_margin - min_x_margin;

        // Width should be smaller with margin applied
        // Without margin: IFP width = 100 - 10 = 90
        // With margin 5: effective boundary is 90x90, IFP width = 90 - 10 = 80
        assert!(
            width_margin < width_no,
            "width_margin ({}) should be < width_no ({})",
            width_margin,
            width_no
        );
    }

    #[test]
    fn test_ifp_margin_boundary_collapse() {
        let boundary = rect(20.0, 20.0);

        // Margin of 12 would make the effective boundary negative (collapse)
        let result = shrink_polygon(&boundary, 12.0);
        assert!(result.is_err(), "Boundary should collapse with margin >= width/2");
    }

    #[test]
    fn test_ifp_margin_large_geometry() {
        let boundary = rect(30.0, 30.0);
        let geom = Geometry2D::rectangle("G", 20.0, 20.0);

        // Without margin: IFP width = 30 - 20 = 10
        let ifp_no_margin = compute_ifp(&boundary, &geom, 0.0).unwrap();
        let (min_x_no, _, max_x_no, _) = ifp_bounding_box(&ifp_no_margin);
        let width_no = max_x_no - min_x_no;

        // With margin 5: effective boundary is 20x20, IFP width = 20 - 20 = 0
        let ifp_with_margin = compute_ifp_with_margin(&boundary, &geom, 0.0, 5.0).unwrap();
        let (min_x_margin, _, max_x_margin, _) = ifp_bounding_box(&ifp_with_margin);
        let width_margin = max_x_margin - min_x_margin;

        // IFP should be smaller (possibly degenerate) with margin
        assert!(
            width_margin <= width_no,
            "width_margin ({}) should be <= width_no ({})",
            width_margin,
            width_no
        );
    }

    #[test]
    fn test_nfp_non_convex_l_shape() {
        // L-shape is not convex
        let l_shape = Geometry2D::new("L")
            .with_polygon(vec![
                (0.0, 0.0),
                (20.0, 0.0),
                (20.0, 10.0),
                (10.0, 10.0),
                (10.0, 20.0),
                (0.0, 20.0),
            ]);

        let small_square = Geometry2D::rectangle("S", 5.0, 5.0);

        // Should compute NFP for non-convex polygon
        let nfp = compute_nfp(&l_shape, &small_square, 0.0).unwrap();

        assert!(!nfp.is_empty());
        // NFP should have multiple vertices due to non-convex shape
        assert!(nfp.vertex_count() >= 4);
    }

    #[test]
    fn test_triangulate_polygon_convex() {
        let square = rect(10.0, 10.0);
        let triangles = triangulate_polygon(&square);

        // Convex polygon should return itself
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0].len(), 4);
    }

    #[test]
    fn test_triangulate_polygon_non_convex() {
        // L-shape
        let l_shape = vec![
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (10.0, 10.0),
            (10.0, 20.0),
            (0.0, 20.0),
        ];

        let triangles = triangulate_polygon(&l_shape);

        // Should triangulate into multiple triangles
        assert!(triangles.len() >= 1);
    }

    #[test]
    fn test_union_polygons() {
        // Two overlapping squares
        let poly1 = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let poly2 = vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)];

        let result = union_polygons(&[poly1, poly2]).unwrap();

        assert!(!result.is_empty());
        // Union of two overlapping squares should have more than 4 vertices
        assert!(result.vertex_count() >= 6);
    }
}
