//! BRKGA-based 3D bin packing optimization.
//!
//! This module provides BRKGA (Biased Random-Key Genetic Algorithm) based
//! optimization for 3D bin packing problems. BRKGA uses random-key encoding
//! and biased crossover to favor elite parents.
//!
//! # Random-Key Encoding
//!
//! Each solution is encoded as a vector of random keys in [0, 1):
//! - First N keys: decoded as permutation (placement order)
//! - Next N keys: decoded as orientation indices
//!
//! # Reference
//!
//! Gonçalves, J. F., & Resende, M. G. (2013). A biased random key genetic
//! algorithm for 2D and 3D bin packing problems.

use crate::boundary::Boundary3D;
use crate::geometry::Geometry3D;
use crate::packing_utils::{
    build_instances, build_unplaced_list, layer_place_items, packing_fitness, InstanceInfo,
    PlacementItem,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::brkga::{BrkgaConfig, BrkgaProblem, BrkgaRunner, RandomKeyChromosome};
use u_nesting_core::solver::Config;
use u_nesting_core::SolveResult;

/// BRKGA problem definition for 3D bin packing.
pub struct BrkgaPackingProblem {
    /// Input geometries.
    geometries: Vec<Geometry3D>,
    /// Boundary container.
    boundary: Boundary3D,
    /// Solver configuration.
    config: Config,
    /// Instance mapping.
    instances: Vec<InstanceInfo>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl BrkgaPackingProblem {
    /// Creates a new BRKGA packing problem.
    pub fn new(
        geometries: Vec<Geometry3D>,
        boundary: Boundary3D,
        config: Config,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        let instances = build_instances(&geometries);

        Self {
            geometries,
            boundary,
            config,
            instances,
            cancelled,
        }
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Decodes a chromosome into placements using layer-based packing.
    ///
    /// The chromosome keys are interpreted as:
    /// - Keys [0..N): placement order (sorted indices)
    /// - Keys [N..2N): orientation indices (discretized)
    pub fn decode(
        &self,
        chromosome: &RandomKeyChromosome,
    ) -> (Vec<u_nesting_core::Placement<f64>>, f64, usize) {
        let n = self.instances.len();
        if n == 0 || chromosome.len() < n {
            return (Vec::new(), 0.0, 0);
        }

        // Decode placement order from first N keys
        let order = chromosome.decode_as_permutation();
        let order: Vec<usize> = order.into_iter().take(n).collect();

        let items: Vec<PlacementItem> = order
            .iter()
            .map(|&instance_idx| {
                let orientation_key_idx = n + instance_idx;
                let orientation_idx = if orientation_key_idx < chromosome.len() {
                    let orient_count = self
                        .instances
                        .get(instance_idx)
                        .map(|i| i.orientation_count)
                        .unwrap_or(1);
                    chromosome.decode_as_discrete(orientation_key_idx, orient_count)
                } else {
                    0
                };
                PlacementItem {
                    instance_idx,
                    orientation_idx,
                }
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

impl BrkgaProblem for BrkgaPackingProblem {
    fn num_keys(&self) -> usize {
        // N keys for order + N keys for orientations
        self.instances.len() * 2
    }

    fn evaluate(&self, chromosome: &mut RandomKeyChromosome) {
        let (_, utilization, placed_count) = self.decode(chromosome);
        let fitness = packing_fitness(placed_count, self.instances.len(), utilization);
        chromosome.set_fitness(fitness);
    }

    fn on_generation(
        &self,
        generation: u32,
        best: &RandomKeyChromosome,
        _population: &[RandomKeyChromosome],
    ) {
        log::debug!(
            "BRKGA 3D Packing Gen {}: fitness={:.4}",
            generation,
            best.fitness()
        );
    }
}

/// Runs BRKGA-based 3D bin packing optimization.
///
/// # Errors
/// `ConfigError` when the algorithm configuration is out of range.
pub fn run_brkga_packing(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    brkga_config: BrkgaConfig,
    cancelled: Arc<AtomicBool>,
) -> u_nesting_core::Result<SolveResult<f64>> {
    let problem = BrkgaPackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = BrkgaRunner::with_cancellation(brkga_config, problem, cancelled.clone())?;

    let brkga_result = runner.run();

    // Decode the best chromosome
    let problem = BrkgaPackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        Arc::new(AtomicBool::new(false)),
    );

    let (placements, utilization, _placed_count) = problem.decode(&brkga_result.best);
    let unplaced = build_unplaced_list(&placements, geometries);

    let mut result = SolveResult::new();
    result.placements = placements;
    result.unplaced = unplaced;
    result.boundaries_used = 1;
    result.utilization = utilization;
    result.computation_time_ms = brkga_result.elapsed.as_millis() as u64;
    result.generations = Some(brkga_result.generations);
    result.best_fitness = Some(brkga_result.best.fitness());
    result.fitness_history = Some(brkga_result.history);
    result.strategy = Some("BRKGA".to_string());
    result.cancelled = cancelled.load(Ordering::Relaxed);
    result.target_reached = brkga_result.target_reached;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brkga_packing_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_brkga_packing(
            &geometries,
            &boundary,
            &config,
            brkga_config,
            Arc::new(AtomicBool::new(false)),
        )
        .expect("valid config");

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("BRKGA".to_string()));
    }

    #[test]
    fn test_brkga_packing_all_placed() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(30);

        let result = run_brkga_packing(
            &geometries,
            &boundary,
            &config,
            brkga_config,
            Arc::new(AtomicBool::new(false)),
        )
        .expect("valid config");

        // All 4 boxes should fit easily
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_brkga_packing_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Long boxes that benefit from rotation
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(3)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(60.0, 60.0, 60.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(30);

        let result = run_brkga_packing(
            &geometries,
            &boundary,
            &config,
            brkga_config,
            Arc::new(AtomicBool::new(false)),
        )
        .expect("valid config");

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_brkga_problem_decode() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // Use a very large boundary with small items to ensure placement always succeeds
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(2)];

        let boundary = Boundary3D::new(500.0, 500.0, 500.0); // Very large boundary
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = BrkgaPackingProblem::new(geometries, boundary, config, cancelled);

        assert_eq!(problem.num_instances(), 2);
        // 2 instances * 2 (order + orientation) = 4 keys
        assert_eq!(problem.num_keys(), 4);

        // Use a seeded RNG for reproducibility
        let mut rng = StdRng::seed_from_u64(42);
        let chromosome = RandomKeyChromosome::random(problem.num_keys(), &mut rng);
        let (placements, utilization, placed_count) = problem.decode(&chromosome);

        // With such a large boundary (500^3 vs 10^3 items), at least one item should fit
        assert!(
            placed_count >= 1,
            "Expected at least 1 placement but got {}",
            placed_count
        );
        assert_eq!(placements.len(), placed_count);
        if placed_count > 0 {
            assert!(utilization > 0.0);
        }
    }

    #[test]
    fn test_brkga_packing_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0).with_max_mass(350.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_brkga_packing(
            &geometries,
            &boundary,
            &config,
            brkga_config,
            Arc::new(AtomicBool::new(false)),
        )
        .expect("valid config");

        // Should only place 3 boxes due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }
}
