//! No-Fit Polygon (NFP) computation.
//!
//! The NFP of two polygons A and B represents all positions where the reference
//! point of B can be placed such that B touches or overlaps A.
//!
//! This module implements:
//! - **Convex case**: Minkowski sum algorithm (O(n+m) for convex polygons)
//! - **Non-convex case**: Convex decomposition + union approach using `i_overlay`
//! - **Sliding algorithm**: Burke et al. (2007) orbiting/sliding approach for robust NFP
//!
//! ## Algorithm Selection
//!
//! Use [`NfpMethod`] to choose the algorithm:
//! - `MinkowskiSum`: Fast for convex polygons, uses decomposition for non-convex
//! - `Sliding`: More robust for complex shapes, follows polygon boundary
//!
//! ```rust,ignore
//! use u_nesting_d2::nfp::{compute_nfp_with_method, NfpMethod};
//!
//! let nfp = compute_nfp_with_method(&stationary, &orbiting, 0.0, NfpMethod::Sliding)?;
//! ```

use crate::geometry::Geometry2D;
use crate::nfp_sliding::{compute_nfp_sliding, SlidingNfpConfig};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};
use u_nesting_core::geom::polygon as geom_polygon;
use u_nesting_core::geometry::Geometry2DExt;
use u_nesting_core::robust::{orient2d_filtered, Orientation};
use u_nesting_core::{Error, Result};

use crate::placement_utils::polygon_centroid;

/// Rotates an NFP around the origin by the given angle (in radians).
///
/// This is used when computing NFP with relative rotation and then
/// transforming it to the actual placed geometry's rotation.
pub fn rotate_nfp(nfp: &Nfp, angle: f64) -> Nfp {
    if angle.abs() < 1e-10 {
        return nfp.clone();
    }

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    Nfp {
        polygons: nfp
            .polygons
            .iter()
            .map(|polygon| {
                polygon
                    .iter()
                    .map(|&(x, y)| (x * cos_a - y * sin_a, x * sin_a + y * cos_a))
                    .collect()
            })
            .collect(),
    }
}

/// Translates an NFP by the given offset.
pub fn translate_nfp(nfp: &Nfp, offset: (f64, f64)) -> Nfp {
    Nfp {
        polygons: nfp
            .polygons
            .iter()
            .map(|polygon| {
                polygon
                    .iter()
                    .map(|(x, y)| (x + offset.0, y + offset.1))
                    .collect()
            })
            .collect(),
    }
}

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

// ============================================================================
// NFP Method Selection
// ============================================================================

/// Method for computing No-Fit Polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NfpMethod {
    /// Minkowski sum algorithm.
    ///
    /// - **Convex polygons**: O(n+m) time complexity
    /// - **Non-convex polygons**: Uses convex decomposition + union
    /// - Best for: Simple shapes, fast computation
    #[default]
    MinkowskiSum,

    /// Sliding/orbiting algorithm (Burke et al. 2007).
    ///
    /// - Traces the NFP boundary by sliding one polygon around another
    /// - More robust for complex interlocking shapes
    /// - Better handles edge cases like perfect fits
    /// - Best for: Complex non-convex shapes, high accuracy requirements
    Sliding,
}

/// Configuration for NFP computation.
#[derive(Debug, Clone)]
pub struct NfpConfig {
    /// The method to use for NFP computation.
    pub method: NfpMethod,
    /// Tolerance for contact detection (Sliding method).
    pub contact_tolerance: f64,
    /// Maximum iterations for sliding algorithm.
    pub max_iterations: usize,
}

impl Default for NfpConfig {
    fn default() -> Self {
        Self {
            method: NfpMethod::MinkowskiSum,
            contact_tolerance: 1e-6,
            max_iterations: 10000,
        }
    }
}

impl NfpConfig {
    /// Creates a new config with the specified method.
    pub fn with_method(method: NfpMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Sets the contact tolerance (for Sliding method).
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.contact_tolerance = tolerance;
        self
    }

    /// Sets the maximum iterations (for Sliding method).
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }
}

/// Computes the No-Fit Polygon between two geometries using the specified method.
///
/// # Arguments
/// * `stationary` - The fixed polygon
/// * `orbiting` - The polygon to be placed
/// * `rotation` - Rotation angle of the orbiting polygon in radians
/// * `method` - The algorithm to use
///
/// # Returns
/// The computed NFP, or an error if computation fails.
pub fn compute_nfp_with_method(
    stationary: &Geometry2D,
    orbiting: &Geometry2D,
    rotation: f64,
    method: NfpMethod,
) -> Result<Nfp> {
    compute_nfp_with_config(
        stationary,
        orbiting,
        rotation,
        &NfpConfig::with_method(method),
    )
}

/// Computes the No-Fit Polygon between two geometries with full configuration.
///
/// # Arguments
/// * `stationary` - The fixed polygon
/// * `orbiting` - The polygon to be placed
/// * `rotation` - Rotation angle of the orbiting polygon in radians
/// * `config` - Configuration including method and parameters
///
/// # Returns
/// The computed NFP, or an error if computation fails.
pub fn compute_nfp_with_config(
    stationary: &Geometry2D,
    orbiting: &Geometry2D,
    rotation: f64,
    config: &NfpConfig,
) -> Result<Nfp> {
    let stat_exterior = stationary.exterior();
    let orb_exterior = orbiting.exterior();

    if stat_exterior.len() < 3 || orb_exterior.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Polygons must have at least 3 vertices".into(),
        ));
    }

    // Apply rotation to orbiting polygon
    let rotated_orbiting = rotate_polygon(orb_exterior, rotation);

    match config.method {
        NfpMethod::MinkowskiSum => {
            // Use existing Minkowski sum implementation
            if stationary.is_convex()
                && is_polygon_convex(&rotated_orbiting)
                && stationary.holes().is_empty()
            {
                compute_nfp_convex(stat_exterior, &rotated_orbiting)
            } else {
                compute_nfp_general(stationary, &rotated_orbiting)
            }
        }
        NfpMethod::Sliding => {
            // Use sliding algorithm
            let sliding_config = SlidingNfpConfig {
                contact_tolerance: config.contact_tolerance,
                max_iterations: config.max_iterations,
                min_translation: config.contact_tolerance * 0.01,
            };

            // Reflect the orbiting polygon (NFP requires -B)
            let reflected: Vec<(f64, f64)> =
                rotated_orbiting.iter().map(|&(x, y)| (-x, -y)).collect();

            compute_nfp_sliding(stat_exterior, &reflected, &sliding_config)
        }
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
pub fn compute_nfp(stationary: &Geometry2D, orbiting: &Geometry2D, rotation: f64) -> Result<Nfp> {
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

    // The IFP (Inner-Fit Polygon) is computed using Minkowski EROSION:
    // IFP = boundary ⊖ geometry = ∩_{g ∈ geometry} (boundary - g)
    //
    // This is the set of all positions p where placing the geometry at p
    // keeps ALL vertices inside the boundary.
    //
    // NOTE: This is different from Minkowski SUM (⊕) which gives UNION not intersection!
    // Previous implementation incorrectly used Minkowski sum.
    compute_minkowski_erosion(&effective_boundary, &rotated_geom)
}

/// Computes Minkowski erosion of boundary by geometry: B ⊖ G = ∩_{g ∈ G} (B - g)
///
/// This gives all positions where placing geometry keeps it entirely inside boundary.
/// For a rectangular boundary and convex geometry, this shrinks the boundary
/// by the geometry's extent in each direction.
fn compute_minkowski_erosion(boundary: &[(f64, f64)], geometry: &[(f64, f64)]) -> Result<Nfp> {
    if boundary.len() < 3 || geometry.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Both boundary and geometry must have at least 3 vertices".into(),
        ));
    }

    // Fast path for rectangular boundary (common case)
    let (b_min_x, b_min_y, b_max_x, b_max_y) = bounding_box(boundary);
    let is_rect = boundary.len() == 4
        && boundary.iter().all(|&(x, y)| {
            ((x - b_min_x).abs() < 1e-10 || (x - b_max_x).abs() < 1e-10)
                && ((y - b_min_y).abs() < 1e-10 || (y - b_max_y).abs() < 1e-10)
        });

    // Get geometry bounding box (AABB of the geometry in its current orientation)
    let (g_min_x, g_min_y, g_max_x, g_max_y) = bounding_box(geometry);

    if is_rect {
        // For rectangular boundary: shrink by geometry extents
        // If geometry reference is at origin and vertices span [g_min, g_max],
        // then placement p is valid iff:
        //   p + g_min_x >= b_min_x  =>  p_x >= b_min_x - g_min_x
        //   p + g_max_x <= b_max_x  =>  p_x <= b_max_x - g_max_x
        //   p + g_min_y >= b_min_y  =>  p_y >= b_min_y - g_min_y
        //   p + g_max_y <= b_max_y  =>  p_y <= b_max_y - g_max_y
        let ifp_min_x = b_min_x - g_min_x;
        let ifp_max_x = b_max_x - g_max_x;
        let ifp_min_y = b_min_y - g_min_y;
        let ifp_max_y = b_max_y - g_max_y;

        // Check if IFP is valid (non-empty)
        if ifp_min_x > ifp_max_x + 1e-10 || ifp_min_y > ifp_max_y + 1e-10 {
            return Err(Error::InvalidGeometry(
                "Geometry too large to fit in boundary".into(),
            ));
        }

        // Clamp to ensure valid rectangle
        let ifp_min_x = ifp_min_x.min(ifp_max_x);
        let ifp_min_y = ifp_min_y.min(ifp_max_y);

        return Ok(Nfp::from_polygon(vec![
            (ifp_min_x, ifp_min_y),
            (ifp_max_x, ifp_min_y),
            (ifp_max_x, ifp_max_y),
            (ifp_min_x, ifp_max_y),
        ]));
    }

    // General case: intersect translated boundaries
    // IFP = ∩_{g ∈ G} (B - g)
    // For each geometry vertex g, translate boundary by -g, then intersect all
    compute_minkowski_erosion_general(boundary, geometry)
}

/// General Minkowski erosion using polygon intersection via i_overlay
fn compute_minkowski_erosion_general(
    boundary: &[(f64, f64)],
    geometry: &[(f64, f64)],
) -> Result<Nfp> {
    if geometry.is_empty() {
        return Ok(Nfp::from_polygon(boundary.to_vec()));
    }

    // Start with boundary translated by first geometry vertex
    let first_g = geometry[0];
    let mut result: Vec<[f64; 2]> = boundary
        .iter()
        .map(|&(x, y)| [x - first_g.0, y - first_g.1])
        .collect();

    // Intersect with boundary translated by each remaining vertex
    for &(gx, gy) in geometry.iter().skip(1) {
        let translated: Vec<[f64; 2]> = boundary.iter().map(|&(x, y)| [x - gx, y - gy]).collect();

        // Intersect current result with translated boundary using i_overlay
        let shapes = result.overlay(&[translated], OverlayRule::Intersect, FillRule::NonZero);

        if shapes.is_empty() {
            return Err(Error::InvalidGeometry(
                "Geometry too large to fit in boundary".into(),
            ));
        }

        // Take the first (largest) resulting polygon
        result = Vec::new();
        for shape in &shapes {
            for contour in shape {
                if contour.len() >= 3 {
                    result = contour.clone();
                    break;
                }
            }
            if !result.is_empty() {
                break;
            }
        }

        if result.len() < 3 {
            return Err(Error::InvalidGeometry(
                "Geometry too large to fit in boundary".into(),
            ));
        }
    }

    // Convert back to (f64, f64) format
    let result_tuples: Vec<(f64, f64)> = result.iter().map(|&[x, y]| (x, y)).collect();
    Ok(Nfp::from_polygon(result_tuples))
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
                return Err(Error::InvalidGeometry("Offset polygon collapsed".into()));
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
        return Err(Error::InvalidGeometry("Offset polygon collapsed".into()));
    }

    // Check if the polygon has positive area (not self-intersecting)
    let area = signed_area(&result).abs();
    if area <= 1e-10 {
        return Err(Error::InvalidGeometry("Offset polygon collapsed".into()));
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

/// Computes NFP for two convex polygons using u-geometry's Minkowski sum.
///
/// Delegates to u-geometry's O(n+m) rotating calipers algorithm.
/// For the NFP, computes: A ⊕ (-B) where ⊕ is Minkowski sum.
fn compute_nfp_convex(stationary: &[(f64, f64)], orbiting: &[(f64, f64)]) -> Result<Nfp> {
    use u_nesting_core::geom::minkowski::nfp_convex;

    let polygon = nfp_convex(stationary, orbiting);
    Ok(Nfp::from_polygon(polygon))
}

/// Computes Minkowski sum of two convex polygons via u-geometry.
///
/// Time complexity: O(n + m) where n, m are vertex counts.
fn compute_minkowski_sum_convex(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> Result<Nfp> {
    use u_nesting_core::geom::minkowski::minkowski_sum_convex;

    let polygon = minkowski_sum_convex(poly_a, poly_b);
    Ok(Nfp::from_polygon(polygon))
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

    // Compute pairwise Minkowski sums in parallel
    // Create all pairs for parallel processing
    let pairs: Vec<_> = stat_triangles
        .iter()
        .flat_map(|stat_tri| {
            orb_triangles
                .iter()
                .map(move |orb_tri| (stat_tri.clone(), orb_tri.clone()))
        })
        .collect();

    let partial_nfps: Vec<Vec<(f64, f64)>> = pairs
        .par_iter()
        .flat_map(|(stat_tri, orb_tri)| {
            // Reflect orbiting triangle
            let reflected: Vec<(f64, f64)> = orb_tri.iter().map(|&(x, y)| (-x, -y)).collect();

            // Compute Minkowski sum of two convex polygons
            if let Ok(nfp) = compute_minkowski_sum_convex(stat_tri, &reflected) {
                nfp.polygons
                    .into_iter()
                    .filter(|polygon| polygon.len() >= 3)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        })
        .collect();

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

/// Checks if a point is strictly inside a triangle using robust predicates.
///
/// Uses robust orientation tests to correctly handle near-degenerate cases.
fn point_in_triangle_robust(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let o1 = orient2d_filtered(a, b, p);
    let o2 = orient2d_filtered(b, c, p);
    let o3 = orient2d_filtered(c, a, p);

    // Point is strictly inside if all orientations are the same
    // (all CCW or all CW) and none are collinear
    (o1 == Orientation::CounterClockwise
        && o2 == Orientation::CounterClockwise
        && o3 == Orientation::CounterClockwise)
        || (o1 == Orientation::Clockwise
            && o2 == Orientation::Clockwise
            && o3 == Orientation::Clockwise)
}

/// Checks if vertex i forms an ear in the polygon.
///
/// Uses robust geometric predicates for numerical stability.
fn is_ear(vertices: &[(f64, f64)], prev: usize, curr: usize, next: usize) -> bool {
    let a = vertices[prev];
    let b = vertices[curr];
    let c = vertices[next];

    // Check if the vertex is convex (turn left in CCW polygon)
    // Using robust orientation test instead of cross product
    let orientation = orient2d_filtered(a, b, c);
    if !orientation.is_ccw() {
        return false; // Reflex or collinear vertex, not an ear
    }

    // Check if any other vertex is inside this triangle
    for (i, &p) in vertices.iter().enumerate() {
        if i == prev || i == curr || i == next {
            continue;
        }
        if point_in_triangle_robust(p, a, b, c) {
            return false;
        }
    }

    true
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
    let mut result: Vec<Vec<[f64; 2]>> = vec![polygons[0].iter().map(|&(x, y)| [x, y]).collect()];

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

/// Checks if a polygon is convex using robust orientation tests.
fn is_polygon_convex(polygon: &[(f64, f64)]) -> bool {
    geom_polygon::is_convex(polygon)
}

/// Ensures polygon vertices are in counter-clockwise order.
fn ensure_ccw(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    geom_polygon::ensure_ccw(polygon)
}

/// Computes the signed area of a polygon.
/// Positive for counter-clockwise, negative for clockwise.
fn signed_area(polygon: &[(f64, f64)]) -> f64 {
    geom_polygon::signed_area(polygon)
}

/// Computes convex hull of a set of points.
fn convex_hull_of_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    geom_polygon::convex_hull(points)
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

/// Checks if a point is strictly outside all NFPs (boundary points are considered outside).
/// This allows pieces to touch but not overlap.
fn point_outside_all_nfps_strict(point: (f64, f64), nfps: &[&Nfp]) -> bool {
    for nfp in nfps {
        for polygon in &nfp.polygons {
            // Point must be strictly outside (interior = overlapping)
            if point_in_polygon(point, polygon) {
                return false;
            }
        }
    }
    true
}

/// Checks if a point is on the boundary of a polygon (not strictly inside or outside).
fn point_on_polygon_boundary(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let (px, py) = point;
    let n = polygon.len();
    const EPS: f64 = 1e-10;

    for i in 0..n {
        let (x1, y1) = polygon[i];
        let (x2, y2) = polygon[(i + 1) % n];

        // Check if point is on this edge segment
        // Using parametric form: P = P1 + t*(P2-P1), 0 <= t <= 1
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        if len_sq < EPS * EPS {
            // Degenerate edge - check if point is at vertex
            if (px - x1).abs() < EPS && (py - y1).abs() < EPS {
                return true;
            }
            continue;
        }

        // Project point onto line
        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;

        // Check if projection is within segment
        if (-EPS..=1.0 + EPS).contains(&t) {
            // Check distance from line
            let proj_x = x1 + t * dx;
            let proj_y = y1 + t * dy;
            let dist_sq = (px - proj_x).powi(2) + (py - proj_y).powi(2);

            if dist_sq < EPS * EPS {
                return true;
            }
        }
    }

    false
}

/// Finds the optimal placement point that minimizes strip length.
///
/// The valid region is defined as points that are:
/// 1. Inside the IFP (Inner-Fit Polygon) - the boundary constraint
/// 2. Outside all NFPs (No-Fit Polygons) - not overlapping placed pieces
///
/// The optimization strategy prioritizes:
/// 1. Minimize X coordinate first (to minimize strip length in strip packing)
/// 2. Then minimize Y coordinate (pack tightly bottom-to-top)
///
/// This approach produces shorter strip lengths than the traditional
/// "bottom-left" approach which prioritizes Y over X.
///
/// # Arguments
/// * `ifp` - The inner-fit polygon (valid positions within boundary)
/// * `nfps` - List of NFPs with already placed pieces
/// * `sample_step` - Grid sampling step size (smaller = more accurate but slower)
///
/// # Returns
/// The optimal valid point, or None if no valid position exists.
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

    // Filter candidates to those inside IFP (including boundary) and outside all NFPs
    let valid_candidates: Vec<(f64, f64)> = candidates
        .into_iter()
        .filter(|&point| {
            // Must be inside IFP (including boundary points)
            let in_ifp = ifp
                .polygons
                .iter()
                .any(|p| point_in_polygon(point, p) || point_on_polygon_boundary(point, p));
            if !in_ifp {
                return false;
            }
            // Must be outside all NFPs (boundary points OK - touching but not overlapping)
            point_outside_all_nfps_strict(point, nfps)
        })
        .collect();

    // Find optimal point: minimize X first (strip length), then Y (pack tightly)
    // This produces shorter strip lengths than the traditional "bottom-left" approach
    valid_candidates.into_iter().min_by(|a, b| {
        // Compare X first (left = shorter strip), then Y (bottom)
        match a.0.partial_cmp(&b.0) {
            Some(std::cmp::Ordering::Equal) => {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
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

/// Verifies that a geometry at the given position does not overlap with any placed geometries.
///
/// This uses actual polygon-polygon intersection testing (SAT) rather than
/// relying solely on NFP point-in-polygon checks, providing more robust
/// collision detection.
///
/// # Arguments
/// * `geometry` - The geometry to be placed
/// * `position` - The position (x, y) for the geometry
/// * `rotation` - The rotation angle in radians
/// * `placed_geometries` - List of already placed geometries
///
/// # Returns
/// `true` if there is NO overlap (placement is valid), `false` if overlap detected
pub fn verify_no_overlap(
    geometry: &Geometry2D,
    position: (f64, f64),
    rotation: f64,
    placed_geometries: &[PlacedGeometry],
) -> bool {
    use crate::nfp_sliding::polygons_overlap;

    // Get the transformed polygon for the geometry being placed
    let rotated = rotate_polygon(geometry.exterior(), rotation);
    let transformed: Vec<(f64, f64)> = rotated
        .into_iter()
        .map(|(x, y)| (x + position.0, y + position.1))
        .collect();

    // Check against each placed geometry
    for placed in placed_geometries {
        let placed_polygon = placed.translated_exterior();

        if polygons_overlap(&transformed, &placed_polygon) {
            return false; // Overlap detected
        }
    }

    true // No overlap
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
        // Valid placements: X in [0, 90], Y in [0, 90]
        let polygon = &ifp.polygons[0];
        let (min_x, min_y, max_x, max_y) = bounding_box(polygon);
        assert_relative_eq!(min_x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(min_y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_x, 90.0, epsilon = 1e-10);
        assert_relative_eq!(max_y, 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ifp_bounds_correct() {
        // Test case from failing test: 25x25 rectangle in 100x50 boundary
        let boundary = rect(100.0, 50.0);
        let geom = Geometry2D::rectangle("R", 25.0, 25.0);

        let ifp = compute_ifp(&boundary, &geom, 0.0).unwrap();

        assert!(!ifp.is_empty());
        // IFP should be [0, 75] x [0, 25]
        let polygon = &ifp.polygons[0];
        let (min_x, min_y, max_x, max_y) = bounding_box(polygon);
        assert_relative_eq!(min_x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(min_y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_x, 75.0, epsilon = 1e-10);
        assert_relative_eq!(max_y, 25.0, epsilon = 1e-10);

        // Positions at (0,0), (25,0), (50,0), (75,0) should all be valid
        assert!(point_in_polygon((0.0, 0.0), polygon) || point_on_boundary((0.0, 0.0), polygon));
        assert!(point_in_polygon((25.0, 0.0), polygon) || point_on_boundary((25.0, 0.0), polygon));
        assert!(point_in_polygon((50.0, 0.0), polygon) || point_on_boundary((50.0, 0.0), polygon));
        assert!(point_in_polygon((75.0, 0.0), polygon) || point_on_boundary((75.0, 0.0), polygon));
    }

    /// Helper to check if point is on polygon boundary
    fn point_on_boundary(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
        let (px, py) = point;
        let n = polygon.len();
        for i in 0..n {
            let (x1, y1) = polygon[i];
            let (x2, y2) = polygon[(i + 1) % n];
            // Check if point is on line segment
            let d1 = ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
            let d2 = ((px - x2).powi(2) + (py - y2).powi(2)).sqrt();
            let d_total = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            if (d1 + d2 - d_total).abs() < 1e-10 {
                return true;
            }
        }
        false
    }

    #[test]
    fn test_nfp_same_size_rectangles() {
        // Two same-size rectangles: NFP should be twice the size
        let a = Geometry2D::rectangle("A", 25.0, 25.0);
        let b = Geometry2D::rectangle("B", 25.0, 25.0);

        let nfp = compute_nfp(&a, &b, 0.0).unwrap();
        assert!(!nfp.is_empty());

        let polygon = &nfp.polygons[0];
        let (min_x, min_y, max_x, max_y) = bounding_box(polygon);
        // NFP should span from -25 to +25 in each dimension = 50x50
        // Actually depends on reference point. Let's check actual dimensions.
        let width = max_x - min_x;
        let height = max_y - min_y;
        eprintln!("NFP dimensions: {}x{}", width, height);
        eprintln!(
            "NFP bounds: ({}, {}) to ({}, {})",
            min_x, min_y, max_x, max_y
        );
        // NFP of two identical rectangles should be 50x50
        assert_relative_eq!(width, 50.0, epsilon = 1e-6);
        assert_relative_eq!(height, 50.0, epsilon = 1e-6);
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
        assert_eq!(compute_count.load(std::sync::atomic::Ordering::SeqCst), 1);
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
        assert!(
            result.is_err(),
            "Polygon should collapse when offset >= width/2"
        );
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
        assert!(
            result.is_err(),
            "Boundary should collapse with margin >= width/2"
        );
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
        let l_shape = Geometry2D::new("L").with_polygon(vec![
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
        assert!(!triangles.is_empty());
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

    // ========================================================================
    // Near-Degenerate Case Tests (Numerical Robustness)
    // ========================================================================

    #[test]
    fn test_convex_near_collinear_vertices() {
        // Polygon with nearly collinear vertices that could fail with naive arithmetic
        let near_collinear = vec![
            (0.0, 0.0),
            (1.0, 1e-15), // Nearly on line y=0
            (2.0, 0.0),
            (2.0, 1.0),
            (0.0, 1.0),
        ];

        // Should handle without crashing
        let result = is_polygon_convex(&near_collinear);
        // The result depends on numerical precision, but it shouldn't panic
        let _ = result; // Just verify it doesn't panic
    }

    #[test]
    fn test_triangulation_near_degenerate() {
        // L-shape with vertices very close together
        let near_degenerate_l = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (5.0 + 1e-12, 5.0), // Very close to (5, 5)
            (5.0, 10.0),
            (0.0, 10.0),
        ];

        // Should triangulate without crashing
        let triangles = triangulate_polygon(&near_degenerate_l);

        // Should produce at least one triangle
        assert!(!triangles.is_empty());
    }

    #[test]
    fn test_nfp_nearly_touching_rectangles() {
        // Two rectangles that are nearly touching (gap of 1e-10)
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        // Should compute NFP correctly even with near-degenerate cases
        let nfp = compute_nfp(&a, &b, 0.0).unwrap();
        assert!(!nfp.is_empty());
    }

    #[test]
    fn test_ifp_geometry_nearly_fills_boundary() {
        // Geometry that nearly fills the boundary (leaves very small margin)
        let boundary = rect(100.0, 100.0);
        let geom = Geometry2D::rectangle("G", 99.9999, 99.9999);

        // Should handle without error
        let result = compute_ifp(&boundary, &geom, 0.0);

        // Either succeeds with a tiny IFP or fails gracefully
        match result {
            Ok(ifp) => {
                // IFP should be very small or a single point
                let (min_x, min_y, max_x, max_y) = ifp_bounding_box(&ifp);
                let width = max_x - min_x;
                let height = max_y - min_y;
                assert!(width < 0.001 && height < 0.001);
            }
            Err(_) => {
                // Also acceptable - geometry too large to fit meaningfully
            }
        }
    }

    #[test]
    fn test_point_in_polygon_on_boundary() {
        // Test point exactly on polygon edge
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];

        // Points on edges
        let on_bottom_edge = (5.0, 0.0);
        let on_right_edge = (10.0, 5.0);
        let on_top_edge = (5.0, 10.0);
        let on_left_edge = (0.0, 5.0);

        // Ray casting algorithm behavior on boundaries is implementation-defined,
        // but it should not crash
        let _ = point_in_polygon(on_bottom_edge, &square);
        let _ = point_in_polygon(on_right_edge, &square);
        let _ = point_in_polygon(on_top_edge, &square);
        let _ = point_in_polygon(on_left_edge, &square);
    }

    #[test]
    fn test_point_in_triangle_robust_degenerate() {
        // Degenerate triangle (all points collinear)
        let a = (0.0, 0.0);
        let b = (5.0, 0.0);
        let c = (10.0, 0.0);

        // Point on the line
        let p = (3.0, 0.0);

        // Should return false (not inside a degenerate triangle)
        assert!(!point_in_triangle_robust(p, a, b, c));
    }

    #[test]
    fn test_ear_detection_with_collinear_points() {
        // Polygon with collinear consecutive vertices
        let with_collinear = vec![
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0), // Collinear with previous two
            (10.0, 10.0),
            (0.0, 10.0),
        ];

        // Should handle triangulation without crashing
        let triangles = triangulate_polygon(&with_collinear);

        // Should produce valid triangles
        for triangle in &triangles {
            assert!(triangle.len() >= 3);
        }
    }

    #[test]
    fn test_nfp_with_very_small_polygon() {
        // Very small polygon (micrometer scale)
        let tiny = Geometry2D::rectangle("tiny", 1e-6, 1e-6);
        let normal = Geometry2D::rectangle("normal", 10.0, 10.0);

        // Should compute NFP correctly
        let nfp = compute_nfp(&normal, &tiny, 0.0).unwrap();
        assert!(!nfp.is_empty());
    }

    #[test]
    fn test_nfp_with_very_large_polygon() {
        // Very large polygon (kilometer scale)
        let large = Geometry2D::rectangle("large", 1e6, 1e6);
        let normal = Geometry2D::rectangle("normal", 100.0, 100.0);

        // Should compute NFP correctly
        let nfp = compute_nfp(&large, &normal, 0.0).unwrap();
        assert!(!nfp.is_empty());
    }

    #[test]
    fn test_signed_area_with_extreme_coordinates() {
        // Polygon with very large coordinates
        // Note: Standard floating-point arithmetic loses precision at extreme magnitudes.
        // This test documents the limitation - for better precision at extreme scales,
        // use the robust::signed_area_robust from u_nesting_core.

        // Moderate scale - should be accurate
        let moderate_coords = vec![
            (1e6, 1e6),
            (1e6 + 100.0, 1e6),
            (1e6 + 100.0, 1e6 + 100.0),
            (1e6, 1e6 + 100.0),
        ];

        let area = signed_area(&moderate_coords);

        // Area should be 10000 (100 * 100)
        assert_relative_eq!(area.abs(), 10000.0, epsilon = 1.0);
    }

    #[test]
    fn test_ensure_ccw_with_near_zero_area() {
        // Polygon with very small area
        let tiny_area = vec![(0.0, 0.0), (1e-10, 0.0), (1e-10, 1e-10), (0.0, 1e-10)];

        // Should handle without crashing
        let ccw = ensure_ccw(&tiny_area);
        assert_eq!(ccw.len(), tiny_area.len());
    }

    // ========================================================================
    // NfpMethod Tests
    // ========================================================================

    #[test]
    fn test_nfp_method_default() {
        let config = NfpConfig::default();
        assert_eq!(config.method, NfpMethod::MinkowskiSum);
    }

    #[test]
    fn test_nfp_method_minkowski_sum() {
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        let nfp = compute_nfp_with_method(&a, &b, 0.0, NfpMethod::MinkowskiSum).unwrap();

        assert!(!nfp.is_empty());
        assert!(nfp.vertex_count() >= 4);
    }

    #[test]
    fn test_nfp_method_sliding() {
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        let result = compute_nfp_with_method(&a, &b, 0.0, NfpMethod::Sliding);

        // Sliding algorithm should return a valid result
        assert!(result.is_ok(), "Sliding method should not error");
        let nfp = result.unwrap();
        assert!(!nfp.is_empty(), "NFP should not be empty");

        // Note: Sliding algorithm is still being improved.
        // For simple convex cases, MinkowskiSum is more reliable.
        // Sliding is intended for complex non-convex cases with interlocking shapes.
    }

    #[test]
    fn test_nfp_method_config_builder() {
        let config = NfpConfig::with_method(NfpMethod::Sliding)
            .with_tolerance(1e-5)
            .with_max_iterations(5000);

        assert_eq!(config.method, NfpMethod::Sliding);
        assert!((config.contact_tolerance - 1e-5).abs() < 1e-10);
        assert_eq!(config.max_iterations, 5000);
    }

    #[test]
    fn test_nfp_methods_both_succeed() {
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        let nfp_mink = compute_nfp_with_method(&a, &b, 0.0, NfpMethod::MinkowskiSum).unwrap();
        let nfp_slide = compute_nfp_with_method(&a, &b, 0.0, NfpMethod::Sliding).unwrap();

        // Both methods should produce non-empty results
        assert!(!nfp_mink.is_empty());
        assert!(!nfp_slide.is_empty());

        // Minkowski sum for convex shapes is well-tested
        assert!(nfp_mink.vertex_count() >= 4);

        // Sliding algorithm produces valid (though possibly different) NFP
        // Note: The sliding algorithm implementation is still being refined.
        // For simple convex cases, both methods should produce geometrically
        // similar results, but vertex count may differ due to different algorithms.
    }

    #[test]
    fn test_nfp_sliding_l_shape() {
        // L-shape (non-convex)
        let l_shape = Geometry2D::new("L").with_polygon(vec![
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (10.0, 10.0),
            (10.0, 20.0),
            (0.0, 20.0),
        ]);

        let small_square = Geometry2D::rectangle("S", 5.0, 5.0);

        // Sliding should handle non-convex shapes without crashing
        let result = compute_nfp_with_method(&l_shape, &small_square, 0.0, NfpMethod::Sliding);

        // Sliding algorithm should at least produce some result for L-shapes
        assert!(result.is_ok(), "Sliding should not error on L-shape");
        let nfp = result.unwrap();
        assert!(!nfp.is_empty(), "NFP should not be empty for L-shape");
    }

    #[test]
    fn test_nfp_with_config() {
        let a = Geometry2D::rectangle("A", 10.0, 10.0);
        let b = Geometry2D::rectangle("B", 5.0, 5.0);

        let config = NfpConfig {
            method: NfpMethod::Sliding,
            contact_tolerance: 1e-4,
            max_iterations: 2000,
        };

        let nfp = compute_nfp_with_config(&a, &b, 0.0, &config).unwrap();

        assert!(!nfp.is_empty());
    }
}
