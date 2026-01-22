//! Exact solver configuration and result types.
//!
//! This module provides types for MILP-based exact solving of small nesting instances.
//! Exact solvers guarantee optimal solutions but are computationally expensive,
//! making them suitable only for small instances (typically ≤15-20 pieces).
//!
//! # Features
//!
//! - `ExactConfig`: Configuration for exact solvers (time limits, gap tolerance)
//! - `ExactResult`: Extended result with optimality proof information
//! - `SolutionStatus`: Optimal, Feasible, Infeasible, or Timeout
//!
//! # Example
//!
//! ```ignore
//! use u_nesting_core::exact::{ExactConfig, SolutionStatus};
//!
//! let config = ExactConfig::default()
//!     .with_time_limit_ms(60000)  // 1 minute
//!     .with_gap_tolerance(0.01);  // 1% optimality gap
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Solution status from exact solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SolutionStatus {
    /// Proven optimal solution found.
    Optimal,
    /// Feasible solution found, but optimality not proven.
    Feasible,
    /// Problem is infeasible (no valid placement exists).
    Infeasible,
    /// Time limit reached without finding any feasible solution.
    Timeout,
    /// Solver encountered an error.
    Error,
    /// Solution status unknown or not applicable.
    #[default]
    Unknown,
}

impl std::fmt::Display for SolutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Optimal => write!(f, "Optimal"),
            Self::Feasible => write!(f, "Feasible"),
            Self::Infeasible => write!(f, "Infeasible"),
            Self::Timeout => write!(f, "Timeout"),
            Self::Error => write!(f, "Error"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Configuration for exact (MILP-based) solvers.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExactConfig {
    /// Maximum computation time in milliseconds.
    pub time_limit_ms: u64,

    /// Relative MIP gap tolerance (0.0 = optimal, 0.01 = 1% gap allowed).
    pub gap_tolerance: f64,

    /// Maximum number of items for exact solving (fallback to heuristic if exceeded).
    pub max_items: usize,

    /// Grid discretization step for position variables.
    pub grid_step: f64,

    /// Number of discrete rotation angles to consider.
    pub rotation_steps: usize,

    /// Whether to use symmetry breaking constraints.
    pub use_symmetry_breaking: bool,

    /// Whether to use valid inequalities (cuts).
    pub use_cuts: bool,

    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: u32,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for ExactConfig {
    fn default() -> Self {
        Self {
            time_limit_ms: 60000, // 1 minute default
            gap_tolerance: 0.0,   // Require optimal
            max_items: 15,        // Small instances only
            grid_step: 1.0,       // 1 unit grid
            rotation_steps: 4,    // 0, 90, 180, 270 degrees
            use_symmetry_breaking: true,
            use_cuts: true,
            verbosity: 0,
            seed: None,
        }
    }
}

impl ExactConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set time limit in milliseconds.
    pub fn with_time_limit_ms(mut self, ms: u64) -> Self {
        self.time_limit_ms = ms;
        self
    }

    /// Set MIP gap tolerance.
    pub fn with_gap_tolerance(mut self, gap: f64) -> Self {
        self.gap_tolerance = gap.clamp(0.0, 1.0);
        self
    }

    /// Set maximum number of items for exact solving.
    pub fn with_max_items(mut self, max: usize) -> Self {
        self.max_items = max.max(1);
        self
    }

    /// Set grid discretization step.
    pub fn with_grid_step(mut self, step: f64) -> Self {
        self.grid_step = step.max(0.1);
        self
    }

    /// Set number of discrete rotation angles.
    pub fn with_rotation_steps(mut self, steps: usize) -> Self {
        self.rotation_steps = steps.max(1);
        self
    }

    /// Enable or disable symmetry breaking constraints.
    pub fn with_symmetry_breaking(mut self, enable: bool) -> Self {
        self.use_symmetry_breaking = enable;
        self
    }

    /// Enable or disable valid inequalities (cuts).
    pub fn with_cuts(mut self, enable: bool) -> Self {
        self.use_cuts = enable;
        self
    }

    /// Set verbosity level.
    pub fn with_verbosity(mut self, level: u32) -> Self {
        self.verbosity = level;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Check if the number of items is within the exact solving limit.
    pub fn is_within_limit(&self, num_items: usize) -> bool {
        num_items <= self.max_items
    }

    /// Get discrete rotation angles in radians.
    pub fn rotation_angles(&self) -> Vec<f64> {
        let step = std::f64::consts::TAU / self.rotation_steps as f64;
        (0..self.rotation_steps).map(|i| i as f64 * step).collect()
    }
}

/// Extended result information from exact solver.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExactResult {
    /// Solution status.
    pub status: SolutionStatus,

    /// Best objective value found (lower is better for minimization).
    pub objective_value: f64,

    /// Best bound on optimal value (for optimality gap calculation).
    pub best_bound: f64,

    /// Optimality gap: (objective - bound) / objective.
    pub gap: f64,

    /// Number of branch-and-bound nodes explored.
    pub nodes_explored: u64,

    /// Number of simplex iterations.
    pub iterations: u64,

    /// Whether the solution is proven optimal.
    pub is_optimal: bool,

    /// Solver-specific status message.
    pub message: String,
}

impl ExactResult {
    /// Create a new result with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a result indicating optimal solution.
    pub fn optimal(objective: f64) -> Self {
        Self {
            status: SolutionStatus::Optimal,
            objective_value: objective,
            best_bound: objective,
            gap: 0.0,
            is_optimal: true,
            message: "Optimal solution found".to_string(),
            ..Default::default()
        }
    }

    /// Create a result indicating feasible (but not proven optimal) solution.
    pub fn feasible(objective: f64, bound: f64) -> Self {
        let gap = if objective.abs() > 1e-10 {
            (objective - bound).abs() / objective.abs()
        } else {
            0.0
        };
        Self {
            status: SolutionStatus::Feasible,
            objective_value: objective,
            best_bound: bound,
            gap,
            is_optimal: false,
            message: format!("Feasible solution found (gap: {:.2}%)", gap * 100.0),
            ..Default::default()
        }
    }

    /// Create a result indicating infeasibility.
    pub fn infeasible() -> Self {
        Self {
            status: SolutionStatus::Infeasible,
            objective_value: f64::INFINITY,
            best_bound: f64::INFINITY,
            is_optimal: false,
            message: "Problem is infeasible".to_string(),
            ..Default::default()
        }
    }

    /// Create a result indicating timeout.
    pub fn timeout(best_objective: Option<f64>, best_bound: f64) -> Self {
        match best_objective {
            Some(obj) => {
                let gap = if obj.abs() > 1e-10 {
                    (obj - best_bound).abs() / obj.abs()
                } else {
                    0.0
                };
                Self {
                    status: SolutionStatus::Timeout,
                    objective_value: obj,
                    best_bound,
                    gap,
                    is_optimal: false,
                    message: format!("Time limit reached (gap: {:.2}%)", gap * 100.0),
                    ..Default::default()
                }
            }
            None => Self {
                status: SolutionStatus::Timeout,
                objective_value: f64::INFINITY,
                best_bound,
                is_optimal: false,
                message: "Time limit reached without feasible solution".to_string(),
                ..Default::default()
            },
        }
    }

    /// Create a result indicating an error.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            status: SolutionStatus::Error,
            message: message.into(),
            ..Default::default()
        }
    }

    /// Set solver statistics.
    pub fn with_stats(mut self, nodes: u64, iterations: u64) -> Self {
        self.nodes_explored = nodes;
        self.iterations = iterations;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_config_default() {
        let config = ExactConfig::default();
        assert_eq!(config.time_limit_ms, 60000);
        assert_eq!(config.gap_tolerance, 0.0);
        assert_eq!(config.max_items, 15);
        assert_eq!(config.rotation_steps, 4);
    }

    #[test]
    fn test_exact_config_builder() {
        let config = ExactConfig::new()
            .with_time_limit_ms(30000)
            .with_gap_tolerance(0.01)
            .with_max_items(10)
            .with_rotation_steps(8)
            .with_grid_step(0.5);

        assert_eq!(config.time_limit_ms, 30000);
        assert_eq!(config.gap_tolerance, 0.01);
        assert_eq!(config.max_items, 10);
        assert_eq!(config.rotation_steps, 8);
        assert_eq!(config.grid_step, 0.5);
    }

    #[test]
    fn test_rotation_angles() {
        let config = ExactConfig::default().with_rotation_steps(4);
        let angles = config.rotation_angles();
        assert_eq!(angles.len(), 4);
        assert!((angles[0] - 0.0).abs() < 1e-10);
        assert!((angles[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((angles[2] - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_is_within_limit() {
        let config = ExactConfig::default().with_max_items(10);
        assert!(config.is_within_limit(5));
        assert!(config.is_within_limit(10));
        assert!(!config.is_within_limit(11));
    }

    #[test]
    fn test_solution_status_display() {
        assert_eq!(format!("{}", SolutionStatus::Optimal), "Optimal");
        assert_eq!(format!("{}", SolutionStatus::Feasible), "Feasible");
        assert_eq!(format!("{}", SolutionStatus::Infeasible), "Infeasible");
        assert_eq!(format!("{}", SolutionStatus::Timeout), "Timeout");
    }

    #[test]
    fn test_exact_result_optimal() {
        let result = ExactResult::optimal(100.0);
        assert_eq!(result.status, SolutionStatus::Optimal);
        assert_eq!(result.objective_value, 100.0);
        assert_eq!(result.gap, 0.0);
        assert!(result.is_optimal);
    }

    #[test]
    fn test_exact_result_feasible() {
        let result = ExactResult::feasible(100.0, 95.0);
        assert_eq!(result.status, SolutionStatus::Feasible);
        assert_eq!(result.objective_value, 100.0);
        assert_eq!(result.best_bound, 95.0);
        assert!((result.gap - 0.05).abs() < 1e-10);
        assert!(!result.is_optimal);
    }

    #[test]
    fn test_exact_result_timeout() {
        let result = ExactResult::timeout(Some(100.0), 90.0);
        assert_eq!(result.status, SolutionStatus::Timeout);
        assert!((result.gap - 0.10).abs() < 1e-10);

        let result_no_solution = ExactResult::timeout(None, 0.0);
        assert_eq!(result_no_solution.status, SolutionStatus::Timeout);
        assert_eq!(result_no_solution.objective_value, f64::INFINITY);
    }

    #[test]
    fn test_exact_result_with_stats() {
        let result = ExactResult::optimal(100.0).with_stats(1000, 50000);
        assert_eq!(result.nodes_explored, 1000);
        assert_eq!(result.iterations, 50000);
    }
}
