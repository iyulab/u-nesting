//! # U-Nesting 2D
//!
//! 2D nesting algorithms for the U-Nesting spatial optimization engine.
//!
//! This crate provides polygon-based 2D nesting with NFP (No-Fit Polygon) computation
//! and various placement algorithms.
//!
//! ## Features
//!
//! - Polygon geometry with holes support
//! - Multiple placement strategies (BLF, NFP-guided, GA, BRKGA, SA)
//! - Convex hull and convexity detection
//! - Configurable rotation and mirroring constraints
//! - NFP-based collision-free placement
//! - Spatial indexing for fast queries
//!
//! ## Quick Start
//!
//! ```rust
//! use u_nesting_d2::{Geometry2D, Boundary2D, Nester2D, Config, Strategy, Solver};
//!
//! // Create geometries
//! let rect = Geometry2D::rectangle("rect1", 100.0, 50.0)
//!     .with_quantity(5)
//!     .with_rotations_deg(vec![0.0, 90.0]);
//!
//! // Create boundary
//! let boundary = Boundary2D::rectangle(500.0, 300.0);
//!
//! // Configure and solve
//! let config = Config::new()
//!     .with_strategy(Strategy::NfpGuided)
//!     .with_spacing(2.0);
//!
//! let nester = Nester2D::new(config);
//! let result = nester.solve(&[rect], &boundary).unwrap();
//!
//! println!("Placed {} items, utilization: {:.1}%",
//!     result.placements.len(),
//!     result.utilization * 100.0);
//! ```
//!
//! ## Geometry Creation
//!
//! ```rust
//! use u_nesting_d2::Geometry2D;
//!
//! // Rectangle
//! let rect = Geometry2D::rectangle("r1", 100.0, 50.0);
//!
//! // Circle (approximated)
//! let circle = Geometry2D::circle("c1", 25.0, 32);
//!
//! // L-shape
//! let l_shape = Geometry2D::l_shape("l1", 100.0, 80.0, 30.0, 30.0);
//!
//! // Custom polygon
//! let custom = Geometry2D::new("custom")
//!     .with_polygon(vec![(0.0, 0.0), (100.0, 0.0), (50.0, 80.0)])
//!     .with_quantity(3);
//! ```

pub mod alns_nesting;
pub mod boundary;
pub mod brkga_nesting;
pub mod ga_nesting;
pub mod gdrr_nesting;
pub mod geometry;
pub mod nester;
pub mod nfp;
#[cfg(feature = "milp")]
pub mod nfp_cm_solver;
pub mod nfp_sliding;
pub mod placement_utils;
pub(crate) mod polygon_ops;
pub mod sa_nesting;
pub mod spatial_index;

/// Computes valid placement bounds and clamps a position to keep geometry within boundary.
///
/// Returns `Some((clamped_x, clamped_y))` if the geometry can fit in the boundary,
/// `None` if the geometry is too large to fit.
///
/// # Arguments
/// * `x`, `y` - The proposed placement position for the geometry's origin
/// * `geom_aabb` - The AABB `(min, max)` of the geometry at the given rotation
/// * `boundary_aabb` - The AABB `(min, max)` of the boundary
pub fn clamp_placement_to_boundary(
    x: f64,
    y: f64,
    geom_aabb: ([f64; 2], [f64; 2]),
    boundary_aabb: ([f64; 2], [f64; 2]),
) -> Option<(f64, f64)> {
    let (g_min, g_max) = geom_aabb;
    let (b_min, b_max) = boundary_aabb;

    // Calculate valid position bounds
    // For geometry to stay inside boundary:
    // - x + g_min[0] >= b_min[0]  => x >= b_min[0] - g_min[0]
    // - x + g_max[0] <= b_max[0]  => x <= b_max[0] - g_max[0]
    let min_valid_x = b_min[0] - g_min[0];
    let max_valid_x = b_max[0] - g_max[0];
    let min_valid_y = b_min[1] - g_min[1];
    let max_valid_y = b_max[1] - g_max[1];

    // Check if geometry can fit
    if max_valid_x < min_valid_x || max_valid_y < min_valid_y {
        // Geometry is too large to fit in boundary
        return None;
    }

    let clamped_x = x.clamp(min_valid_x, max_valid_x);
    let clamped_y = y.clamp(min_valid_y, max_valid_y);

    Some((clamped_x, clamped_y))
}

/// Computes valid placement bounds with margin and clamps a position to keep geometry within boundary.
///
/// Returns `Some((clamped_x, clamped_y))` if the geometry can fit in the boundary (with margin),
/// `None` if the geometry is too large to fit.
///
/// # Arguments
/// * `x`, `y` - The proposed placement position for the geometry's origin
/// * `geom_aabb` - The AABB `(min, max)` of the geometry at the given rotation
/// * `boundary_aabb` - The AABB `(min, max)` of the boundary
/// * `margin` - The margin to apply inside the boundary
pub fn clamp_placement_to_boundary_with_margin(
    x: f64,
    y: f64,
    geom_aabb: ([f64; 2], [f64; 2]),
    boundary_aabb: ([f64; 2], [f64; 2]),
    margin: f64,
) -> Option<(f64, f64)> {
    let (g_min, g_max) = geom_aabb;
    let (b_min, b_max) = boundary_aabb;

    // Calculate valid position bounds (with margin applied to effective boundary)
    // Effective boundary: [b_min + margin, b_max - margin]
    // For geometry to stay inside effective boundary:
    // - x + g_min[0] >= b_min[0] + margin  => x >= b_min[0] + margin - g_min[0]
    // - x + g_max[0] <= b_max[0] - margin  => x <= b_max[0] - margin - g_max[0]
    let min_valid_x = b_min[0] + margin - g_min[0];
    let max_valid_x = b_max[0] - margin - g_max[0];
    let min_valid_y = b_min[1] + margin - g_min[1];
    let max_valid_y = b_max[1] - margin - g_max[1];

    // Check if geometry can fit
    if max_valid_x < min_valid_x || max_valid_y < min_valid_y {
        // Geometry is too large to fit in boundary with the given margin
        return None;
    }

    let clamped_x = x.clamp(min_valid_x, max_valid_x);
    let clamped_y = y.clamp(min_valid_y, max_valid_y);

    Some((clamped_x, clamped_y))
}

/// Checks if a placement lies within the boundary, at least `margin` from its
/// edges (exterior and holes).
///
/// Returns `true` if the geometry at the given placement — rotated, and
/// mirrored when the placement says so — is fully within the boundary and no
/// closer than `margin` to any boundary edge, `false` otherwise.
///
/// # Arguments
/// * `placement` - The placement to validate (position, rotation, mirroring)
/// * `geometry` - The geometry being placed
/// * `boundary` - The boundary to check against
/// * `margin` - Minimum distance from the piece to every boundary edge
/// * `tolerance` - Small tolerance for floating point comparison (e.g., 1e-6)
pub fn is_placement_within_bounds(
    placement: &Placement<f64>,
    geometry: &Geometry2D,
    boundary: &Boundary2D,
    margin: f64,
    tolerance: f64,
) -> bool {
    use u_nesting_core::geometry::Boundary;
    use u_nesting_core::Boundary2DExt;

    let x = placement.position.first().copied().unwrap_or(0.0);
    let y = placement.position.get(1).copied().unwrap_or(0.0);
    let rotation = placement.rotation.first().copied().unwrap_or(0.0);

    let (g_min, g_max) = geometry.aabb_at_rotation_mirrored(rotation, placement.mirrored);
    let (b_min, b_max) = boundary.aabb();

    // AABB containment within the margin — a necessary condition, and *exact*
    // for a hole-free axis-aligned rectangular boundary.
    let aabb_inside = x + g_min[0] >= b_min[0] + margin - tolerance
        && x + g_max[0] <= b_max[0] - margin + tolerance
        && y + g_min[1] >= b_min[1] + margin - tolerance
        && y + g_max[1] <= b_max[1] - margin + tolerance;

    // For a plain rectangle (width & height set, no holes) the AABB check is the
    // exact answer. Infinite strips must also stay on the AABB path: their
    // exterior carries `f64::MAX` vertices, so ray-cast polygon containment is
    // meaningless. Everything else — an arbitrary boundary polygon, or a
    // rectangle carrying holes — needs true polygon-in-polygon containment,
    // because the AABB of a triangular/concave/holed boundary spans empty
    // regions where a piece would sit fully inside the box yet outside the shape.
    let plain_rectangle =
        boundary.width().is_some() && boundary.height().is_some() && boundary.holes().is_empty();
    if boundary.is_infinite() || plain_rectangle {
        return aabb_inside;
    }
    if !aabb_inside {
        return false;
    }

    let base = if placement.mirrored {
        Geometry2D::new(geometry.id().clone())
            .with_polygon(polygon_ops::mirror_polygon(geometry.exterior()))
    } else {
        geometry.clone()
    };
    let piece = base.transformed_exterior(x, y, rotation);
    if !boundary.contains_polygon(&piece) {
        return false;
    }
    margin <= 0.0
        || std::iter::once(boundary.exterior())
            .chain(boundary.holes().iter().map(Vec::as_slice))
            .all(|ring| polygon_ops::ring_distance(&piece, ring) >= margin - tolerance)
}

/// Validates all placements in a SolveResult and removes any that are outside the boundary.
///
/// Returns a new SolveResult with only valid placements, updated utilization,
/// and invalid placements added to the unplaced list.
///
/// # Arguments
/// * `result` - The solve result to validate
/// * `geometries` - The geometries that were being placed
/// * `boundary` - The boundary to check against
/// * `margin` - Minimum distance a placement must keep from the boundary edges
pub fn validate_and_filter_placements(
    mut result: SolveResult<f64>,
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    margin: f64,
) -> SolveResult<f64> {
    use std::collections::HashMap;
    use u_nesting_core::geometry::{Boundary, Geometry};

    const TOLERANCE: f64 = 1e-6;

    // Build a map from geometry ID to geometry for quick lookup
    let geom_map: HashMap<_, _> = geometries.iter().map(|g| (g.id().clone(), g)).collect();

    let (b_min, b_max) = boundary.aabb();
    log::debug!(
        "Validating placements against boundary: ({:.2}, {:.2}) to ({:.2}, {:.2})",
        b_min[0],
        b_min[1],
        b_max[0],
        b_max[1]
    );

    let mut valid_placements = Vec::new();
    let mut total_valid_area = 0.0;
    let mut filtered_count = 0;
    // Used-footprint accumulator: the AABB union of the placed pieces, which is
    // boundary-padding independent (unlike `utilization`, which divides by the
    // full boundary and shrinks arbitrarily as boundary height grows).
    let mut used_min_x = f64::INFINITY;
    let mut used_min_y = f64::INFINITY;
    let mut used_max_x = f64::NEG_INFINITY;
    let mut used_max_y = f64::NEG_INFINITY;

    for placement in result.placements {
        if let Some(geom) = geom_map.get(&placement.geometry_id) {
            let px = placement.position.first().copied().unwrap_or(0.0);
            let py = placement.position.get(1).copied().unwrap_or(0.0);
            let rot = placement.rotation.first().copied().unwrap_or(0.0);

            if is_placement_within_bounds(&placement, geom, boundary, margin, TOLERANCE) {
                total_valid_area += geom.measure();
                let (pg_min, pg_max) = geom.aabb_at_rotation_mirrored(rot, placement.mirrored);
                used_min_x = used_min_x.min(px + pg_min[0]);
                used_min_y = used_min_y.min(py + pg_min[1]);
                used_max_x = used_max_x.max(px + pg_max[0]);
                used_max_y = used_max_y.max(py + pg_max[1]);
                valid_placements.push(placement);
            } else {
                // Calculate actual bounds for debugging
                let (g_min, g_max) = geom.aabb_at_rotation(rot);
                let placed_min_x = px + g_min[0];
                let placed_max_x = px + g_max[0];
                let placed_min_y = py + g_min[1];
                let placed_max_y = py + g_max[1];

                log::warn!(
                    "FILTERED: {} at ({:.2}, {:.2}) rot={:.2}° - bounds ({:.2}, {:.2}) to ({:.2}, {:.2}) outside boundary",
                    placement.geometry_id,
                    px, py,
                    rot.to_degrees(),
                    placed_min_x, placed_min_y, placed_max_x, placed_max_y
                );
                filtered_count += 1;
                // Add to unplaced list
                result.unplaced.push(placement.geometry_id.clone());
            }
        } else {
            // Geometry not found - shouldn't happen but handle gracefully
            log::warn!("Geometry {} not found in lookup map", placement.geometry_id);
            result.unplaced.push(placement.geometry_id.clone());
        }
    }

    if filtered_count > 0 {
        log::warn!(
            "Validation filtered out {} placements as out-of-bounds",
            filtered_count
        );
    }

    // Update result with valid placements only
    result.placements = valid_placements;
    result.utilization = total_valid_area / boundary.measure();

    // Record the used-footprint metrics (padding-independent). `total_piece_area`
    // is otherwise only populated on the multi-strip path; set it here so the
    // single-sheet `used_utilization = piece_area / used_bbox_area` is meaningful.
    result.total_piece_area = total_valid_area;
    if result.placements.is_empty() {
        result.used_bounding_box = [0.0, 0.0];
    } else {
        result.used_bounding_box = [used_max_x - used_min_x, used_max_y - used_min_y];
    }

    result
}

// Re-exports
pub use boundary::Boundary2D;
pub use geometry::Geometry2D;
pub use nester::Nester2D;
pub use nfp::{NfpConfig, NfpMethod};
pub use spatial_index::{SpatialEntry2D, SpatialIndex2D};
pub use u_nesting_core::{
    Boundary, Boundary2DExt, Config, Error, Geometry, Geometry2DExt, Placement, Result,
    RotationConstraint, SolveResult, Solver, Strategy, Transform2D, AABB2D,
};
