//! # U-Nesting
//!
//! Domain-agnostic 2D/3D spatial optimization engine.
//!
//! This crate provides algorithms for:
//! - **2D Nesting**: Polygon placement optimization (cutting, packing)
//! - **3D Bin Packing**: Box placement in containers
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use u_nesting::d2::{Nester2D, Geometry2D, Boundary2D};
//! use u_nesting::core::{Solver, Config, Strategy};
//!
//! // Create geometries and boundary
//! let boundary = Boundary2D::new(1000.0, 500.0);
//! let geometries = vec![/* your polygons */];
//!
//! // Solve
//! let mut nester = Nester2D::new(boundary, geometries);
//! let result = nester.solve(&Config::default())?;
//! ```
//!
//! ## Feature Flags
//!
//! - `d2` (default): 2D nesting algorithms
//! - `d3` (default): 3D bin packing algorithms
//! - `serde`: Serialization support

/// Core traits and abstractions.
pub use u_nesting_core as core;

/// 2D nesting algorithms.
#[cfg(feature = "d2")]
pub use u_nesting_d2 as d2;

/// 3D bin packing algorithms.
#[cfg(feature = "d3")]
pub use u_nesting_d3 as d3;

/// Cutting path optimization.
#[cfg(feature = "cutting")]
pub use u_nesting_cutting as cutting;

// Re-export commonly used types at root level
pub use u_nesting_core::{Config, Placement, SolveResult, Solver, Strategy};

// The README's Rust examples are the first code most users copy, so they are
// compiled and run with the doc-tests. Without this they were checked by
// nothing.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDoctests;
