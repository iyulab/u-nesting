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
#[cfg(feature = "milp")]
pub mod milp_solver;
pub mod nester;
pub mod nfp;
#[cfg(feature = "milp")]
pub mod nfp_cm_solver;
pub mod nfp_sliding;
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

/// Checks if a placement is within the boundary.
///
/// Returns `true` if the geometry at the given placement is fully within the boundary,
/// `false` otherwise.
///
/// # Arguments
/// * `placement` - The placement to validate (contains position and rotation)
/// * `geometry` - The geometry being placed
/// * `boundary` - The boundary to check against
/// * `tolerance` - Small tolerance for floating point comparison (e.g., 1e-6)
pub fn is_placement_within_bounds(
    placement: &Placement<f64>,
    geometry: &Geometry2D,
    boundary: &Boundary2D,
    tolerance: f64,
) -> bool {
    use u_nesting_core::geometry::Boundary;

    // Extract position (Vec<f64> with [x, y] for 2D)
    let x = placement.position.first().copied().unwrap_or(0.0);
    let y = placement.position.get(1).copied().unwrap_or(0.0);

    // Extract rotation (Vec<f64> with [θ] for 2D)
    let rotation = placement.rotation.first().copied().unwrap_or(0.0);

    // Get geometry AABB at the placement rotation
    let (g_min, g_max) = geometry.aabb_at_rotation(rotation);

    // Get boundary AABB
    let (b_min, b_max) = boundary.aabb();

    // Calculate the actual bounds of the placed geometry
    let placed_min_x = x + g_min[0];
    let placed_max_x = x + g_max[0];
    let placed_min_y = y + g_min[1];
    let placed_max_y = y + g_max[1];

    // Check if fully within boundary (with tolerance)
    placed_min_x >= b_min[0] - tolerance
        && placed_max_x <= b_max[0] + tolerance
        && placed_min_y >= b_min[1] - tolerance
        && placed_max_y <= b_max[1] + tolerance
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
pub fn validate_and_filter_placements(
    mut result: SolveResult<f64>,
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
) -> SolveResult<f64> {
    use std::collections::HashMap;
    use u_nesting_core::geometry::{Boundary, Geometry};

    const TOLERANCE: f64 = 1e-6;

    // Build a map from geometry ID to geometry for quick lookup
    let geom_map: HashMap<_, _> = geometries.iter().map(|g| (g.id().clone(), g)).collect();

    let (b_min, b_max) = boundary.aabb();
    log::debug!(
        "Validating placements against boundary: ({:.2}, {:.2}) to ({:.2}, {:.2})",
        b_min[0], b_min[1], b_max[0], b_max[1]
    );

    let mut valid_placements = Vec::new();
    let mut total_valid_area = 0.0;
    let mut filtered_count = 0;

    for placement in result.placements {
        if let Some(geom) = geom_map.get(&placement.geometry_id) {
            let px = placement.position.first().copied().unwrap_or(0.0);
            let py = placement.position.get(1).copied().unwrap_or(0.0);
            let rot = placement.rotation.first().copied().unwrap_or(0.0);

            if is_placement_within_bounds(&placement, geom, boundary, TOLERANCE) {
                total_valid_area += geom.measure();
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
        log::warn!("Validation filtered out {} placements as out-of-bounds", filtered_count);
    }

    // Update result with valid placements only
    result.placements = valid_placements;
    result.utilization = total_valid_area / boundary.measure();

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
