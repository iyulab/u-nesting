//! Cutting path optimization for 2D nesting results.
//!
//! Given a nesting solve result (placed parts with positions and rotations),
//! this module computes an optimized cutting path that minimizes:
//! - Non-cutting (rapid) travel distance
//! - Number of pierce points
//! - Heat accumulation (optional)
//!
//! # Algorithm
//!
//! 1. **Contour extraction**: Extract cut contours from placed geometries
//! 2. **Hierarchy construction**: Build a DAG of precedence constraints
//!    (holes before exteriors, inner parts before outer parts)
//! 3. **Pierce point selection**: Choose optimal entry points on each contour
//! 4. **Sequence optimization**: TSP-based ordering with precedence constraints
//!    (Nearest Neighbor + constrained 2-opt)
//! 5. **Path assembly**: Combine into final cutting path with rapid moves
//!
//! # References
//!
//! - Dewil et al. (2016), "A review of cutting path algorithms for laser cutters"
//! - Hu, Lin & Fu (2023), "Optimizing Cutting Sequences for Common-Edge Nested Parts"

pub mod config;
pub mod contour;
pub mod cost;
pub mod hierarchy;
pub mod kerf;
pub mod path;
pub mod pierce;
pub mod result;
pub mod sequence;

pub use config::CuttingConfig;
pub use path::optimize_cutting_path;
pub use result::{CutDirection, CutStep, CuttingPathResult};
