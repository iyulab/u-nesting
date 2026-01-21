//! Solve result representation.

use crate::geometry::GeometryId;
use crate::placement::{Placement, PlacementStats};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of a nesting or packing solve operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveResult<S> {
    /// List of placements for all successfully placed geometry instances.
    pub placements: Vec<Placement<S>>,

    /// Number of boundaries (bins) used.
    pub boundaries_used: usize,

    /// Utilization ratio (0.0 - 1.0).
    /// Calculated as: total_geometry_measure / total_boundary_measure
    pub utilization: f64,

    /// IDs of geometries that could not be placed.
    pub unplaced: Vec<GeometryId>,

    /// Computation time in milliseconds.
    pub computation_time_ms: u64,

    /// Number of generations (for GA-based solvers).
    pub generations: Option<u32>,

    /// Number of iterations (for SA-based solvers).
    pub iterations: Option<u64>,

    /// Best fitness value achieved (for GA/SA-based solvers).
    pub best_fitness: Option<f64>,

    /// Fitness history over generations (for analysis).
    pub fitness_history: Option<Vec<f64>>,

    /// Strategy used for solving.
    pub strategy: Option<String>,

    /// Whether the solve was cancelled early.
    pub cancelled: bool,

    /// Whether the target utilization was reached.
    pub target_reached: bool,
}

impl<S> SolveResult<S> {
    /// Creates a new empty result.
    pub fn new() -> Self {
        Self {
            placements: Vec::new(),
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: Vec::new(),
            computation_time_ms: 0,
            generations: None,
            iterations: None,
            best_fitness: None,
            fitness_history: None,
            strategy: None,
            cancelled: false,
            target_reached: false,
        }
    }

    /// Returns true if all geometries were placed.
    pub fn all_placed(&self) -> bool {
        self.unplaced.is_empty()
    }

    /// Returns the number of placed geometry instances.
    pub fn placed_count(&self) -> usize {
        self.placements.len()
    }

    /// Returns the number of unplaced geometry types.
    pub fn unplaced_count(&self) -> usize {
        self.unplaced.len()
    }

    /// Returns true if the solve was successful (at least one placement).
    pub fn is_successful(&self) -> bool {
        !self.placements.is_empty()
    }

    /// Returns true if the solve completed within the time limit.
    pub fn completed_normally(&self) -> bool {
        !self.cancelled
    }

    /// Sets the strategy name.
    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = Some(strategy.into());
        self
    }

    /// Sets the generations count.
    pub fn with_generations(mut self, generations: u32) -> Self {
        self.generations = Some(generations);
        self
    }

    /// Sets the best fitness.
    pub fn with_best_fitness(mut self, fitness: f64) -> Self {
        self.best_fitness = Some(fitness);
        self
    }

    /// Sets the fitness history.
    pub fn with_fitness_history(mut self, history: Vec<f64>) -> Self {
        self.fitness_history = Some(history);
        self
    }

    /// Removes duplicate entries from the unplaced list.
    /// This is useful when multiple instances of the same geometry failed to place.
    pub fn deduplicate_unplaced(&mut self) {
        let mut seen = std::collections::HashSet::new();
        self.unplaced.retain(|id| seen.insert(id.clone()));
    }

    /// Computes placement statistics.
    pub fn placement_stats(&self) -> PlacementStats {
        PlacementStats::from_placements(&self.placements)
    }

    /// Returns utilization as a percentage string.
    pub fn utilization_percent(&self) -> String {
        format!("{:.1}%", self.utilization * 100.0)
    }

    /// Merges placements from another result (for multi-bin scenarios).
    pub fn merge(&mut self, other: SolveResult<S>, boundary_offset: usize) {
        // Offset boundary indices
        for mut placement in other.placements {
            placement.boundary_index += boundary_offset;
            self.placements.push(placement);
        }

        self.boundaries_used = self
            .boundaries_used
            .max(other.boundaries_used + boundary_offset);
        self.unplaced.extend(other.unplaced);
        self.computation_time_ms += other.computation_time_ms;

        // Recalculate utilization would require knowing the total measures
    }
}

impl<S> Default for SolveResult<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a solve result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveSummary {
    /// Total geometries requested.
    pub total_requested: usize,
    /// Total geometries placed.
    pub total_placed: usize,
    /// Utilization percentage.
    pub utilization_percent: f64,
    /// Number of bins/boundaries used.
    pub bins_used: usize,
    /// Computation time in milliseconds.
    pub time_ms: u64,
    /// Strategy used.
    pub strategy: String,
}

impl<S> From<&SolveResult<S>> for SolveSummary {
    fn from(result: &SolveResult<S>) -> Self {
        Self {
            total_requested: result.placements.len() + result.unplaced.len(),
            total_placed: result.placements.len(),
            utilization_percent: result.utilization * 100.0,
            bins_used: result.boundaries_used,
            time_ms: result.computation_time_ms,
            strategy: result
                .strategy
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_new() {
        let result: SolveResult<f64> = SolveResult::new();
        assert!(result.placements.is_empty());
        assert_eq!(result.utilization, 0.0);
        assert!(result.all_placed());
    }

    #[test]
    fn test_result_with_placements() {
        let mut result: SolveResult<f64> = SolveResult::new();
        result
            .placements
            .push(Placement::new_2d("test".to_string(), 0, 0.0, 0.0, 0.0));
        result.utilization = 0.85;

        assert_eq!(result.placed_count(), 1);
        assert!(result.is_successful());
        assert_eq!(result.utilization_percent(), "85.0%");
    }

    #[test]
    fn test_result_with_unplaced() {
        let mut result: SolveResult<f64> = SolveResult::new();
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G2".to_string());

        assert!(!result.all_placed());
        assert_eq!(result.unplaced_count(), 2);
    }

    #[test]
    fn test_solve_summary() {
        let mut result: SolveResult<f64> = SolveResult::new();
        result
            .placements
            .push(Placement::new_2d("test".to_string(), 0, 0.0, 0.0, 0.0));
        result.utilization = 0.75;
        result.boundaries_used = 1;
        result.computation_time_ms = 100;
        result.strategy = Some("GA".to_string());

        let summary = SolveSummary::from(&result);
        assert_eq!(summary.total_placed, 1);
        assert_eq!(summary.utilization_percent, 75.0);
        assert_eq!(summary.strategy, "GA");
    }

    #[test]
    fn test_deduplicate_unplaced() {
        let mut result: SolveResult<f64> = SolveResult::new();
        // Simulate multiple instances of same geometry failing to place
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G2".to_string());
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G2".to_string());

        assert_eq!(result.unplaced.len(), 5);

        result.deduplicate_unplaced();

        assert_eq!(result.unplaced.len(), 2);
        assert!(result.unplaced.contains(&"G1".to_string()));
        assert!(result.unplaced.contains(&"G2".to_string()));
    }
}
