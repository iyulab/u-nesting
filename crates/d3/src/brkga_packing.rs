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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::brkga::{BrkgaConfig, BrkgaProblem, BrkgaRunner, RandomKeyChromosome};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

/// Instance information for decoding.
#[derive(Debug, Clone)]
struct InstanceInfo {
    /// Index into the geometries array.
    geometry_idx: usize,
    /// Instance number within this geometry's quantity.
    instance_num: usize,
    /// Number of allowed orientations.
    orientation_count: usize,
}

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
        // Build instance mapping
        let mut instances = Vec::new();

        for (geom_idx, geom) in geometries.iter().enumerate() {
            let orient_count = geom.allowed_orientations().len();

            for instance_num in 0..geom.quantity() {
                instances.push(InstanceInfo {
                    geometry_idx: geom_idx,
                    instance_num,
                    orientation_count: orient_count,
                });
            }
        }

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
    pub fn decode(&self, chromosome: &RandomKeyChromosome) -> (Vec<Placement<f64>>, f64, usize) {
        let n = self.instances.len();
        if n == 0 || chromosome.len() < n {
            return (Vec::new(), 0.0, 0);
        }

        // Decode placement order from first N keys
        let order = chromosome.decode_as_permutation();
        // Only take first N indices
        let order: Vec<usize> = order.into_iter().take(n).collect();

        let mut placements = Vec::new();

        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_max_x = self.boundary.width() - margin;
        let bound_max_y = self.boundary.depth() - margin;
        let bound_max_z = self.boundary.height() - margin;

        // Track current position in layer-based packing
        let mut current_x = margin;
        let mut current_y = margin;
        let mut current_z = margin;
        let mut row_depth = 0.0_f64;
        let mut layer_height = 0.0_f64;

        let mut total_placed_volume = 0.0;
        let mut total_placed_mass = 0.0;
        let mut placed_count = 0;

        // Place items in the decoded order
        for &instance_idx in &order {
            if self.cancelled.load(Ordering::Relaxed) {
                break;
            }

            if instance_idx >= self.instances.len() {
                continue;
            }

            let info = &self.instances[instance_idx];
            let geom = &self.geometries[info.geometry_idx];

            // Decode orientation from the second half of keys
            let orientation_key_idx = n + instance_idx;
            let orientation_idx = if orientation_key_idx < chromosome.len() {
                chromosome.decode_as_discrete(orientation_key_idx, info.orientation_count)
            } else {
                0
            };

            // Get dimensions for this orientation
            let dims = geom.dimensions_for_orientation(orientation_idx);
            let g_width = dims.x;
            let g_depth = dims.y;
            let g_height = dims.z;

            // Check mass constraint
            if let (Some(max_mass), Some(item_mass)) = (self.boundary.max_mass(), geom.mass()) {
                if total_placed_mass + item_mass > max_mass {
                    continue;
                }
            }

            // Try to fit in current row
            if current_x + g_width > bound_max_x {
                // Move to next row
                current_x = margin;
                current_y += row_depth + spacing;
                row_depth = 0.0;
            }

            // Check if fits in current layer (y direction)
            if current_y + g_depth > bound_max_y {
                // Move to next layer
                current_x = margin;
                current_y = margin;
                current_z += layer_height + spacing;
                row_depth = 0.0;
                layer_height = 0.0;
            }

            // Check if fits in container height
            if current_z + g_height > bound_max_z {
                continue;
            }

            // Place the item
            let placement = Placement::new_3d(
                geom.id().clone(),
                info.instance_num,
                current_x,
                current_y,
                current_z,
                0.0,
                0.0,
                0.0, // Orientation is encoded in orientation_idx, not Euler angles
            );

            placements.push(placement);
            total_placed_volume += geom.measure();
            if let Some(mass) = geom.mass() {
                total_placed_mass += mass;
            }
            placed_count += 1;

            // Update position for next item
            current_x += g_width + spacing;
            row_depth = row_depth.max(g_depth);
            layer_height = layer_height.max(g_height);
        }

        let utilization = total_placed_volume / self.boundary.measure();
        (placements, utilization, placed_count)
    }
}

impl BrkgaProblem for BrkgaPackingProblem {
    fn num_keys(&self) -> usize {
        // N keys for order + N keys for orientations
        self.instances.len() * 2
    }

    fn evaluate(&self, chromosome: &mut RandomKeyChromosome) {
        let total_instances = self.instances.len();
        let (_, utilization, placed_count) = self.decode(chromosome);

        // Fitness = placement ratio * 100 + utilization * 10
        let placement_ratio = placed_count as f64 / total_instances.max(1) as f64;
        let fitness = placement_ratio * 100.0 + utilization * 10.0;

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
pub fn run_brkga_packing(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    brkga_config: BrkgaConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let problem = BrkgaPackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = BrkgaRunner::with_cancellation(brkga_config, problem, cancelled.clone());

    let brkga_result = runner.run();

    // Decode the best chromosome
    let problem = BrkgaPackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        Arc::new(AtomicBool::new(false)),
    );

    let (placements, utilization, _placed_count) = problem.decode(&brkga_result.best);

    // Build unplaced list
    let mut unplaced = Vec::new();
    let mut placed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &placements {
        placed_ids.insert(p.geometry_id.clone());
    }
    for geom in geometries {
        if !placed_ids.contains(geom.id()) {
            unplaced.push(geom.id().clone());
        }
    }

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

    result
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
        );

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
        );

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
        );

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
        );

        // Should only place 3 boxes due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }
}
