//! # U-Nesting Core
//!
//! Core traits and abstractions for the U-Nesting spatial optimization engine.
//!
//! This crate provides the foundational types and traits that are shared between
//! the 2D nesting and 3D bin packing modules.
//!
//! ## Core Components
//!
//! - **Geometry traits**: [`Geometry`], [`Geometry2DExt`], [`Geometry3DExt`]
//! - **Boundary traits**: [`Boundary`], [`Boundary2DExt`], [`Boundary3DExt`]
//! - **Solver trait**: [`Solver`] - Common interface for all optimization algorithms
//! - **GA framework**: [`GaRunner`], [`GaProblem`] - Genetic algorithm infrastructure
//! - **BRKGA framework**: [`BrkgaRunner`], [`BrkgaProblem`] - Biased Random-Key GA
//! - **SA framework**: [`SaRunner`], [`SaProblem`] - Simulated Annealing
//! - **Transform types**: [`Transform2D`], [`Transform3D`], [`AABB2D`], [`AABB3D`]
//!
//! ## Optimization Strategies
//!
//! The [`Strategy`] enum defines available optimization algorithms:
//!
//! | Strategy | Speed | Quality | Description |
//! |----------|-------|---------|-------------|
//! | `BottomLeftFill` | Fast | Basic | Greedy bottom-left placement |
//! | `NfpGuided` | Medium | Good | NFP-based optimal positioning (2D) |
//! | `GeneticAlgorithm` | Slow | High | GA with permutation encoding |
//! | `Brkga` | Medium | High | Biased Random-Key GA |
//! | `SimulatedAnnealing` | Medium | High | Temperature-based optimization |
//! | `ExtremePoint` | Fast | Good | EP heuristic (3D only) |
//!
//! ## Configuration
//!
//! Use [`Config`] to configure solver behavior:
//!
//! ```rust
//! use u_nesting_core::{Config, Strategy};
//!
//! let config = Config::new()
//!     .with_strategy(Strategy::GeneticAlgorithm)
//!     .with_spacing(2.0)
//!     .with_margin(5.0)
//!     .with_time_limit(30000);
//! ```
//!
//! ## Feature Flags
//!
//! - `serde`: Enable serialization/deserialization support

pub mod alns;
pub mod brkga;
pub mod error;
pub mod exact;
pub mod ga;
pub mod gdrr;
pub mod geometry;
pub mod memory;
pub mod placement;
pub mod result;
pub mod robust;
pub mod sa;
pub mod solver;
pub mod transform;

// Re-exports
pub use alns::{
    AlnsConfig, AlnsProblem, AlnsProgress, AlnsResult, AlnsRunner, AlnsSolution, DestroyOperatorId,
    DestroyResult, OperatorStats, RepairOperatorId, RepairResult,
};
pub use brkga::{
    BrkgaConfig, BrkgaProblem, BrkgaProgress, BrkgaResult, BrkgaRunner, RandomKeyChromosome,
};
pub use error::{Error, Result};
pub use exact::{ExactConfig, ExactResult, SolutionStatus};
pub use ga::{
    GaConfig, GaProblem, GaProgress, GaResult, GaRunner, Individual, PermutationChromosome,
};
pub use gdrr::{
    GdrrConfig, GdrrProblem, GdrrProgress, GdrrResult, GdrrRunner, GdrrSolution, RecreateResult,
    RecreateType, RuinResult, RuinType, RuinedItem,
};
pub use geometry::{
    Boundary, Boundary2DExt, Boundary3DExt, Geometry, Geometry2DExt, Geometry3DExt, GeometryId,
    Orientation3D, RotationConstraint,
};
pub use placement::Placement;
pub use result::{SolveResult, SolveSummary};
pub use sa::{
    CoolingSchedule, NeighborhoodOperator, PermutationSolution, SaConfig, SaProblem, SaResult,
    SaRunner, SaSolution,
};
pub use solver::{Config, ProgressCallback, ProgressInfo, Solver, Strategy};
pub use transform::{Transform2D, Transform3D, AABB2D, AABB3D};
