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
//! ```rust
//! use u_nesting::d2::{Boundary2D, Geometry2D, Nester2D};
//! use u_nesting::{Config, Solver, Strategy};
//!
//! let geometries = vec![Geometry2D::rectangle("plate", 100.0, 50.0).with_quantity(4)];
//! let boundary = Boundary2D::rectangle(1000.0, 500.0);
//!
//! let config = Config::new().with_strategy(Strategy::NfpGuided);
//! let result = Nester2D::new(config).solve(&geometries, &boundary)?;
//! assert_eq!(result.placements.len(), 4);
//! # Ok::<(), u_nesting::core::Error>(())
//! ```
//!
//! ## Feature Flags
//!
//! - `d2` (default): 2D nesting algorithms
//! - `d3` (default): 3D bin packing algorithms
//! - `cutting`: cutting path optimization for a 2D layout
//! - `serde`: Serialization support
//! - `parallel`: evaluates a search's population (GA, BRKGA) and computes
//!   no-fit polygons on rayon's thread pool. Without it every solve runs on
//!   the calling thread. The solve uses the pool it is called from, so
//!   `rayon::ThreadPool::install` (or `RAYON_NUM_THREADS`) sets how many
//!   threads it uses.

/// This crate's version, as published (`CARGO_PKG_VERSION`) — for a consumer
/// that reports what it was built against without reading a lock file.
///
/// ```
/// assert!(!u_nesting::VERSION.is_empty());
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
pub use u_nesting_core::{Config, Placement, SolveResult, Solver, StopReason, Strategy};

// The README's Rust examples are the first code most users copy, so they are
// compiled and run with the doc-tests. Without this they were checked by
// nothing.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDoctests;
