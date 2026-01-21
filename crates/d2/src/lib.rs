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

pub mod boundary;
pub mod brkga_nesting;
pub mod ga_nesting;
pub mod geometry;
pub mod nester;
pub mod nfp;
pub mod sa_nesting;
pub mod spatial_index;

// Re-exports
pub use boundary::Boundary2D;
pub use geometry::Geometry2D;
pub use nester::Nester2D;
pub use spatial_index::{SpatialEntry2D, SpatialIndex2D};
pub use u_nesting_core::{
    Boundary, Boundary2DExt, Config, Error, Geometry, Geometry2DExt, Placement, Result,
    RotationConstraint, SolveResult, Solver, Strategy, Transform2D, AABB2D,
};
