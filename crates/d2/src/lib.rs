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
pub mod nfp_sliding;
pub mod sa_nesting;
pub mod spatial_index;

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
