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
