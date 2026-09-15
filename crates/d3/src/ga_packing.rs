//! Genetic Algorithm based 3D bin packing optimization.
//!
//! This module provides GA-based optimization for 3D bin packing problems,
//! using permutation + orientation chromosome representation and layer-based decoding.

use crate::boundary::Boundary3D;
use crate::geometry::Geometry3D;
use crate::packing_utils::{
    build_instances, build_unplaced_list, layer_place_items, packing_fitness, InstanceInfo,
    PlacementItem,
};
use rand::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::ga::{GaConfig, GaProblem, GaRunner, Individual};
use u_nesting_core::solver::Config;
use u_nesting_core::SolveResult;

/// Packing chromosome representing placement order and orientations.
#[derive(Debug, Clone)]
pub struct PackingChromosome {
    /// Permutation of instance indices (placement order).
    pub order: Vec<usize>,
    /// Orientation index for each instance (0-5 for boxes with Any constraint).
    pub orientations: Vec<usize>,
    /// Cached fitness value.
    fitness: f64,
    /// Number of placed pieces.
    placed_count: usize,
    /// Total instances count.
    total_count: usize,
}

impl PackingChromosome {
    /// Creates a new chromosome for the given number of instances.
    pub fn new(num_instances: usize) -> Self {
        Self {
            order: (0..num_instances).collect(),
            orientations: vec![0; num_instances],
            fitness: f64::NEG_INFINITY,
            placed_count: 0,
            total_count: num_instances,
        }
    }

    /// Creates a random chromosome with given orientation options per instance.
    pub fn random_with_options<R: Rng>(
        num_instances: usize,
        orientation_counts: &[usize],
        rng: &mut R,
    ) -> Self {
        let mut order: Vec<usize> = (0..num_instances).collect();
        order.shuffle(rng);

        let orientations: Vec<usize> = orientation_counts
            .iter()
            .map(|&count| rng.random_range(0..count.max(1)))
            .collect();

        Self {
            order,
            orientations,
            fitness: f64::NEG_INFINITY,
            placed_count: 0,
            total_count: num_instances,
        }
    }

    /// Sets the fitness value.
    pub fn set_fitness(&mut self, fitness: f64, placed_count: usize) {
        self.fitness = fitness;
        self.placed_count = placed_count;
    }

    /// Order crossover (OX1) for permutation genes.
    pub fn order_crossover<R: Rng>(
        &self,
        other: &Self,
        orientation_counts: &[usize],
        rng: &mut R,
    ) -> Self {
        let n = self.order.len();
        if n < 2 {
            return self.clone();
        }

        // Select two crossover points
        let (mut p1, mut p2) = (rng.random_range(0..n), rng.random_range(0..n));
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        // Copy segment from parent1
        let mut child_order = vec![usize::MAX; n];
        let mut used = vec![false; n];

        for i in p1..=p2 {
            child_order[i] = self.order[i];
            used[self.order[i]] = true;
        }

        // Fill remaining from parent2
        let mut j = (p2 + 1) % n;
        for i in 0..n {
            let idx = (p2 + 1 + i) % n;
            if child_order[idx] == usize::MAX {
                while used[other.order[j]] {
                    j = (j + 1) % n;
                }
                child_order[idx] = other.order[j];
                used[other.order[j]] = true;
                j = (j + 1) % n;
            }
        }

        // Crossover orientations (uniform with bounds checking)
        let orientations: Vec<usize> = self
            .orientations
            .iter()
            .zip(&other.orientations)
            .enumerate()
            .map(|(i, (a, b))| {
                let max_orient = orientation_counts.get(i).copied().unwrap_or(1);
                let chosen = if rng.random() { *a } else { *b };
                chosen % max_orient.max(1)
            })
            .collect();

        Self {
            order: child_order,
            orientations,
            fitness: f64::NEG_INFINITY,
            placed_count: 0,
            total_count: self.total_count,
        }
    }

    /// Swap mutation for order genes.
    pub fn swap_mutate<R: Rng>(&mut self, rng: &mut R) {
        if self.order.len() < 2 {
            return;
        }

        let i = rng.random_range(0..self.order.len());
        let j = rng.random_range(0..self.order.len());
        self.order.swap(i, j);
        self.fitness = f64::NEG_INFINITY;
    }

    /// Orientation mutation.
    pub fn orientation_mutate<R: Rng>(&mut self, orientation_counts: &[usize], rng: &mut R) {
        if self.orientations.is_empty() {
            return;
        }

        let idx = rng.random_range(0..self.orientations.len());
        let max_orient = orientation_counts.get(idx).copied().unwrap_or(1);
        if max_orient > 1 {
            self.orientations[idx] = rng.random_range(0..max_orient);
            self.fitness = f64::NEG_INFINITY;
        }
    }

    /// Inversion mutation (reverses a segment of the order).
    pub fn inversion_mutate<R: Rng>(&mut self, rng: &mut R) {
        let n = self.order.len();
        if n < 2 {
            return;
        }

        let (mut p1, mut p2) = (rng.random_range(0..n), rng.random_range(0..n));
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        self.order[p1..=p2].reverse();
        self.fitness = f64::NEG_INFINITY;
    }
}

impl Individual for PackingChromosome {
    type Fitness = f64;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn random<R: Rng>(_rng: &mut R) -> Self {
        // Default: empty, will be overridden by problem's initialize_population
        Self::new(0)
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        // Default crossover without orientation bounds - should use problem's crossover
        let fake_counts = vec![6; self.orientations.len()];
        self.order_crossover(other, &fake_counts, rng)
    }

    fn mutate<R: Rng>(&mut self, rng: &mut R) {
        // 50% swap, 30% inversion, 20% orientation
        let r: f64 = rng.random();
        if r < 0.5 {
            self.swap_mutate(rng);
        } else if r < 0.8 {
            self.inversion_mutate(rng);
        } else {
            // Orientation mutation with max 6 options
            let fake_counts = vec![6; self.orientations.len()];
            self.orientation_mutate(&fake_counts, rng);
        }
    }
}

/// Problem definition for GA-based 3D bin packing.
pub struct PackingProblem {
    /// Input geometries.
    geometries: Vec<Geometry3D>,
    /// Boundary container.
    boundary: Boundary3D,
    /// Solver configuration.
    config: Config,
    /// Instance mapping.
    instances: Vec<InstanceInfo>,
    /// Orientation counts per instance.
    orientation_counts: Vec<usize>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl PackingProblem {
    /// Creates a new packing problem.
    pub fn new(
        geometries: Vec<Geometry3D>,
        boundary: Boundary3D,
        config: Config,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        let instances = build_instances(&geometries);
        let orientation_counts: Vec<usize> =
            instances.iter().map(|i| i.orientation_count).collect();

        Self {
            geometries,
            boundary,
            config,
            instances,
            orientation_counts,
            cancelled,
        }
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Returns orientation counts per instance.
    pub fn orientation_counts(&self) -> &[usize] {
        &self.orientation_counts
    }

    /// Decodes a chromosome into placements using layer-based packing.
    pub fn decode(
        &self,
        chromosome: &PackingChromosome,
    ) -> (Vec<u_nesting_core::Placement<f64>>, f64, usize) {
        let items: Vec<PlacementItem> = chromosome
            .order
            .iter()
            .map(|&idx| PlacementItem {
                instance_idx: idx,
                orientation_idx: chromosome.orientations.get(idx).copied().unwrap_or(0),
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

impl GaProblem for PackingProblem {
    type Individual = PackingChromosome;

    fn evaluate(&self, individual: &mut Self::Individual) {
        let (_, utilization, placed_count) = self.decode(individual);
        let fitness = packing_fitness(placed_count, individual.total_count, utilization);
        individual.set_fitness(fitness, placed_count);
    }

    fn initialize_population<R: Rng>(&self, size: usize, rng: &mut R) -> Vec<Self::Individual> {
        (0..size)
            .map(|_| {
                PackingChromosome::random_with_options(
                    self.num_instances(),
                    &self.orientation_counts,
                    rng,
                )
            })
            .collect()
    }

    fn on_generation(
        &self,
        generation: u32,
        best: &Self::Individual,
        _population: &[Self::Individual],
    ) {
        log::debug!(
            "GA 3D Packing Gen {}: fitness={:.4}, placed={}/{}",
            generation,
            best.fitness(),
            best.placed_count,
            best.total_count
        );
    }
}

/// Runs GA-based 3D bin packing optimization.
pub fn run_ga_packing(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    ga_config: GaConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let problem = PackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = GaRunner::with_cancellation(ga_config, problem, cancelled.clone());

    let ga_result = runner.run();

    // Decode the best chromosome
    let problem = PackingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        Arc::new(AtomicBool::new(false)),
    );

    let (placements, utilization, _placed_count) = problem.decode(&ga_result.best);
    let unplaced = build_unplaced_list(&placements, geometries);

    let mut result = SolveResult::new();
    result.placements = placements;
    result.unplaced = unplaced;
    result.boundaries_used = 1;
    result.utilization = utilization;
    result.computation_time_ms = ga_result.elapsed.as_millis() as u64;
    result.generations = Some(ga_result.generations);
    result.best_fitness = Some(ga_result.best.fitness());
    result.fitness_history = Some(ga_result.history);
    result.strategy = Some("GeneticAlgorithm".to_string());
    result.cancelled = cancelled.load(Ordering::Relaxed);
    result.target_reached = ga_result.target_reached;

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_chromosome_crossover() {
        let mut rng = rand::rng();
        let orientation_counts = vec![6, 6, 6, 6, 6];
        let parent1 = PackingChromosome::random_with_options(5, &orientation_counts, &mut rng);
        let parent2 = PackingChromosome::random_with_options(5, &orientation_counts, &mut rng);

        let child = parent1.order_crossover(&parent2, &orientation_counts, &mut rng);

        // Child should be a valid permutation
        assert_eq!(child.order.len(), 5);
        let mut sorted = child.order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..5).collect::<Vec<_>>());

        // Orientations should be within bounds
        for (i, &orient) in child.orientations.iter().enumerate() {
            assert!(orient < orientation_counts[i]);
        }
    }

    #[test]
    fn test_packing_chromosome_mutation() {
        let mut rng = rand::rng();
        let orientation_counts = vec![6, 6, 6, 6, 6];
        let mut chromosome =
            PackingChromosome::random_with_options(5, &orientation_counts, &mut rng);

        chromosome.swap_mutate(&mut rng);

        // Should still be a valid permutation
        let mut sorted = chromosome.order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn test_ga_packing_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10);

        let result = run_ga_packing(
            &geometries,
            &boundary,
            &config,
            ga_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_ga_packing_all_placed() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_ga_packing(
            &geometries,
            &boundary,
            &config,
            ga_config,
            Arc::new(AtomicBool::new(false)),
        );

        // All 4 boxes should fit easily
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ga_packing_with_orientations() {
        use crate::geometry::OrientationConstraint;
        // Long boxes that benefit from rotation
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(3)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(60.0, 60.0, 60.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_ga_packing(
            &geometries,
            &boundary,
            &config,
            ga_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_packing_problem_decode() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = PackingProblem::new(geometries, boundary, config, cancelled);

        assert_eq!(problem.num_instances(), 2);

        // Create a chromosome and decode
        let chromosome = PackingChromosome::new(2);
        let (placements, utilization, placed_count) = problem.decode(&chromosome);

        assert_eq!(placed_count, 2);
        assert_eq!(placements.len(), 2);
        assert!(utilization > 0.0);
    }

    #[test]
    fn test_ga_packing_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0).with_max_mass(350.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10);

        let result = run_ga_packing(
            &geometries,
            &boundary,
            &config,
            ga_config,
            Arc::new(AtomicBool::new(false)),
        );

        // Should only place 3 boxes due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }
}
