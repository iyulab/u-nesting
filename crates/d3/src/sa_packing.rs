//! Simulated Annealing-based 3D bin packing optimization.
//!
//! This module provides Simulated Annealing based optimization for 3D bin packing
//! problems. SA uses neighborhood operators to explore the solution space
//! and accepts worse solutions with a probability that decreases over time.
//!
//! # Neighborhood Operators
//!
//! - **Swap**: Exchange positions of two items in the sequence
//! - **Relocate**: Move an item to a different position
//! - **Inversion**: Reverse a segment of the sequence
//! - **Rotation**: Change the orientation of an item

use crate::boundary::Boundary3D;
use crate::geometry::Geometry3D;
use crate::packing_utils::{
    build_instances, build_unplaced_list, layer_place_items, packing_fitness, InstanceInfo,
    PlacementItem,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::sa::{
    NeighborhoodOperator, PermutationSolution, SaConfig, SaProblem, SaRunner, SaSolution,
};
use u_nesting_core::solver::Config;
use u_nesting_core::SolveResult;

/// SA problem definition for 3D bin packing.
pub struct SaPackingProblem {
    /// Input geometries.
    geometries: Vec<Geometry3D>,
    /// Boundary container.
    boundary: Boundary3D,
    /// Solver configuration.
    config: Config,
    /// Instance mapping.
    instances: Vec<InstanceInfo>,
    /// Maximum orientation count across all geometries.
    max_orientation_count: usize,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl SaPackingProblem {
    /// Creates a new SA packing problem.
    pub fn new(
        geometries: Vec<Geometry3D>,
        boundary: Boundary3D,
        config: Config,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        let instances = build_instances(&geometries);
        let max_orientation_count = instances
            .iter()
            .map(|i| i.orientation_count)
            .max()
            .unwrap_or(1);

        Self {
            geometries,
            boundary,
            config,
            instances,
            max_orientation_count,
            cancelled,
        }
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Decodes a solution into placements using layer-based packing.
    pub fn decode(
        &self,
        solution: &PermutationSolution,
    ) -> (Vec<u_nesting_core::Placement<f64>>, f64, usize) {
        let n = self.instances.len();
        if n == 0 || solution.sequence.is_empty() {
            return (Vec::new(), 0.0, 0);
        }

        let items: Vec<PlacementItem> = solution
            .sequence
            .iter()
            .enumerate()
            .map(|(seq_idx, &instance_idx)| PlacementItem {
                instance_idx,
                orientation_idx: solution.rotations.get(seq_idx).copied().unwrap_or(0),
            })
            .collect();

        let result = layer_place_items(
            &items,
            &self.instances,
            &self.geometries,
            &self.boundary,
            &self.config,
            &self.cancelled,
        );
        (result.placements, result.utilization, result.placed_count)
    }
}

impl SaProblem for SaPackingProblem {
    type Solution = PermutationSolution;

    fn initial_solution<R: rand::Rng>(&self, rng: &mut R) -> Self::Solution {
        PermutationSolution::random(self.instances.len(), self.max_orientation_count, rng)
    }

    fn neighbor<R: rand::Rng>(
        &self,
        solution: &Self::Solution,
        operator: NeighborhoodOperator,
        rng: &mut R,
    ) -> Self::Solution {
        match operator {
            NeighborhoodOperator::Swap => solution.apply_swap(rng),
            NeighborhoodOperator::Relocate => solution.apply_relocate(rng),
            NeighborhoodOperator::Inversion => solution.apply_inversion(rng),
            NeighborhoodOperator::Rotation => solution.apply_rotation(rng),
            NeighborhoodOperator::Chain => solution.apply_chain(rng),
            // 3D packing has no `allow_flip`/mirror concept (2D-only, see
            // `NeighborhoodOperator::MirrorFlip` doc comment) — never
            // offered by `available_operators()` below, so unreachable in
            // practice; the arm exists only to satisfy match exhaustiveness.
            NeighborhoodOperator::MirrorFlip => solution.clone(),
        }
    }

    fn evaluate(&self, solution: &mut Self::Solution) {
        let (_, utilization, placed_count) = self.decode(solution);
        let fitness = packing_fitness(placed_count, self.instances.len(), utilization);
        solution.set_objective(fitness);
    }

    fn available_operators(&self) -> Vec<NeighborhoodOperator> {
        if self.max_orientation_count > 1 {
            vec![
                NeighborhoodOperator::Swap,
                NeighborhoodOperator::Relocate,
                NeighborhoodOperator::Inversion,
                NeighborhoodOperator::Rotation,
                NeighborhoodOperator::Chain,
            ]
        } else {
            vec![
                NeighborhoodOperator::Swap,
                NeighborhoodOperator::Relocate,
                NeighborhoodOperator::Inversion,
                NeighborhoodOperator::Chain,
            ]
        }
    }

    fn on_temperature_change(
        &self,
        temperature: f64,
        iteration: u64,
        best: &Self::Solution,
        _current: &Self::Solution,
    ) {
        log::debug!(
            "SA 3D Packing Iteration {}: temp={:.4}, best_fitness={:.4}",
            iteration,
            temperature,
            best.objective()
        );
    }
}

/// Runs SA-based 3D bin packing optimization.
pub fn run_sa_packing(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    sa_config: SaConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let problem = SaPackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = SaRunner::new(sa_config, problem);

    // Set cancellation (thread-based polling, not available on WASM)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cancel_handle = runner.cancel_handle();
        let cancelled_clone = cancelled.clone();
        std::thread::spawn(move || {
            while !cancelled_clone.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            cancel_handle.store(true, Ordering::Relaxed);
        });
    }

    let sa_result = runner.run();

    // Decode the best solution
    let problem = SaPackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        Arc::new(AtomicBool::new(false)),
    );

    let (placements, utilization, _placed_count) = problem.decode(&sa_result.best);
    let unplaced = build_unplaced_list(&placements, geometries);

    let mut result = SolveResult::new();
    result.placements = placements;
    result.unplaced = unplaced;
    result.boundaries_used = 1;
    result.utilization = utilization;
    result.computation_time_ms = sa_result.elapsed.as_millis() as u64;
    result.iterations = Some(sa_result.iterations);
    result.best_fitness = Some(sa_result.best.objective());
    result.fitness_history = Some(sa_result.history);
    result.strategy = Some("SimulatedAnnealing".to_string());
    result.cancelled = cancelled.load(Ordering::Relaxed);
    result.target_reached = sa_result.target_reached;

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sa_packing_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.9)
            .with_iterations_per_temp(20)
            .with_max_iterations(500);

        let result = run_sa_packing(
            &geometries,
            &boundary,
            &config,
            sa_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("SimulatedAnnealing".to_string()));
    }

    #[test]
    fn test_sa_packing_all_placed() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_max_iterations(1000);

        let result = run_sa_packing(
            &geometries,
            &boundary,
            &config,
            sa_config,
            Arc::new(AtomicBool::new(false)),
        );

        // All 4 boxes should fit easily
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_sa_packing_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Long boxes that benefit from rotation
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(3)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(60.0, 60.0, 60.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_max_iterations(500);

        let result = run_sa_packing(
            &geometries,
            &boundary,
            &config,
            sa_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_sa_problem_decode() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = SaPackingProblem::new(geometries, boundary, config, cancelled);

        assert_eq!(problem.num_instances(), 2);

        // Create a random solution and decode
        let mut rng = rand::rng();
        let solution = PermutationSolution::random(problem.num_instances(), 1, &mut rng);
        let (placements, utilization, placed_count) = problem.decode(&solution);

        // Should place at least one item
        assert!(placed_count >= 1);
        assert_eq!(placements.len(), placed_count);
        if placed_count > 0 {
            assert!(utilization > 0.0);
        }
    }

    #[test]
    fn test_sa_packing_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0).with_max_mass(350.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_max_iterations(500);

        let result = run_sa_packing(
            &geometries,
            &boundary,
            &config,
            sa_config,
            Arc::new(AtomicBool::new(false)),
        );

        // Should only place 3 boxes due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }
}
