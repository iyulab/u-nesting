//! Genetic Algorithm based 2D nesting optimization.
//!
//! This module provides GA-based optimization for 2D nesting problems,
//! using the permutation chromosome representation and NFP-guided decoding.

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
use crate::nfp::{compute_ifp, compute_nfp, find_bottom_left_placement, Nfp, PlacedGeometry};
use rand::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::ga::{GaConfig, GaProblem, GaRunner, Individual};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

/// Nesting chromosome representing a placement order and rotations.
#[derive(Debug, Clone)]
pub struct NestingChromosome {
    /// Permutation of geometry indices (placement order).
    pub order: Vec<usize>,
    /// Rotation index for each geometry instance.
    pub rotations: Vec<usize>,
    /// Cached fitness value.
    fitness: f64,
    /// Number of placed pieces (for fitness calculation).
    placed_count: usize,
    /// Total instances count.
    total_count: usize,
}

impl NestingChromosome {
    /// Creates a new chromosome for the given number of instances and rotation options.
    pub fn new(num_instances: usize, _rotation_options: usize) -> Self {
        Self {
            order: (0..num_instances).collect(),
            rotations: vec![0; num_instances],
            fitness: f64::NEG_INFINITY,
            placed_count: 0,
            total_count: num_instances,
        }
    }

    /// Creates a random chromosome.
    pub fn random_with_options<R: Rng>(
        num_instances: usize,
        rotation_options: usize,
        rng: &mut R,
    ) -> Self {
        let mut order: Vec<usize> = (0..num_instances).collect();
        order.shuffle(rng);

        let rotations: Vec<usize> = (0..num_instances)
            .map(|_| rng.gen_range(0..rotation_options.max(1)))
            .collect();

        Self {
            order,
            rotations,
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
    pub fn order_crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let n = self.order.len();
        if n < 2 {
            return self.clone();
        }

        // Select two crossover points
        let (mut p1, mut p2) = (rng.gen_range(0..n), rng.gen_range(0..n));
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

        // Crossover rotations (uniform)
        let rotations: Vec<usize> = self
            .rotations
            .iter()
            .zip(&other.rotations)
            .map(|(a, b)| if rng.gen() { *a } else { *b })
            .collect();

        Self {
            order: child_order,
            rotations,
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

        let i = rng.gen_range(0..self.order.len());
        let j = rng.gen_range(0..self.order.len());
        self.order.swap(i, j);
        self.fitness = f64::NEG_INFINITY;
    }

    /// Rotation mutation.
    pub fn rotation_mutate<R: Rng>(&mut self, rotation_options: usize, rng: &mut R) {
        if self.rotations.is_empty() || rotation_options <= 1 {
            return;
        }

        let idx = rng.gen_range(0..self.rotations.len());
        self.rotations[idx] = rng.gen_range(0..rotation_options);
        self.fitness = f64::NEG_INFINITY;
    }

    /// Inversion mutation (reverses a segment).
    pub fn inversion_mutate<R: Rng>(&mut self, rng: &mut R) {
        let n = self.order.len();
        if n < 2 {
            return;
        }

        let (mut p1, mut p2) = (rng.gen_range(0..n), rng.gen_range(0..n));
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        self.order[p1..=p2].reverse();
        self.fitness = f64::NEG_INFINITY;
    }
}

impl Individual for NestingChromosome {
    type Fitness = f64;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        // Default: empty, will be overridden by problem's initialize_population
        Self::random_with_options(0, 1, rng)
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        self.order_crossover(other, rng)
    }

    fn mutate<R: Rng>(&mut self, rng: &mut R) {
        // 50% swap, 30% inversion, 20% rotation
        let r: f64 = rng.gen();
        if r < 0.5 {
            self.swap_mutate(rng);
        } else if r < 0.8 {
            self.inversion_mutate(rng);
        } else {
            // Rotation mutation with 4 options (0, 90, 180, 270 degrees)
            self.rotation_mutate(4, rng);
        }
    }
}

/// Instance information for decoding.
#[derive(Debug, Clone)]
struct InstanceInfo {
    /// Index into the geometries array.
    geometry_idx: usize,
    /// Instance number within this geometry's quantity.
    instance_num: usize,
}

/// Problem definition for GA-based 2D nesting.
pub struct NestingProblem {
    /// Input geometries.
    geometries: Vec<Geometry2D>,
    /// Boundary container.
    boundary: Boundary2D,
    /// Solver configuration.
    config: Config,
    /// Instance mapping (instance_id -> (geometry_idx, instance_num)).
    instances: Vec<InstanceInfo>,
    /// Available rotation angles per geometry.
    rotation_angles: Vec<Vec<f64>>,
    /// Number of rotation options.
    rotation_options: usize,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl NestingProblem {
    /// Creates a new nesting problem.
    pub fn new(
        geometries: Vec<Geometry2D>,
        boundary: Boundary2D,
        config: Config,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        // Build instance mapping
        let mut instances = Vec::new();
        let mut rotation_angles = Vec::new();

        for (geom_idx, geom) in geometries.iter().enumerate() {
            // Get rotation angles for this geometry
            let angles = geom.rotations();
            let angles = if angles.is_empty() {
                vec![0.0]
            } else {
                angles
            };
            rotation_angles.push(angles);

            // Create instances
            for instance_num in 0..geom.quantity() {
                instances.push(InstanceInfo {
                    geometry_idx: geom_idx,
                    instance_num,
                });
            }
        }

        // Maximum rotation options across all geometries
        let rotation_options = rotation_angles.iter().map(|a| a.len()).max().unwrap_or(1);

        Self {
            geometries,
            boundary,
            config,
            instances,
            rotation_angles,
            rotation_options,
            cancelled,
        }
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Returns the number of rotation options.
    pub fn rotation_options(&self) -> usize {
        self.rotation_options
    }

    /// Decodes a chromosome into placements using NFP-guided placement.
    pub fn decode(&self, chromosome: &NestingChromosome) -> (Vec<Placement<f64>>, f64, usize) {
        let mut placements = Vec::new();
        let mut placed_geometries: Vec<PlacedGeometry> = Vec::new();
        let mut total_placed_area = 0.0;
        let mut placed_count = 0;

        let margin = self.config.margin;
        let spacing = self.config.spacing;

        // Get boundary polygon with margin
        let boundary_polygon = self.get_boundary_polygon_with_margin(margin);

        // Sampling step for grid search
        let sample_step = self.compute_sample_step();

        // Place geometries in the order specified by chromosome
        for &instance_idx in &chromosome.order {
            if self.cancelled.load(Ordering::Relaxed) {
                break;
            }

            if instance_idx >= self.instances.len() {
                continue;
            }

            let info = &self.instances[instance_idx];
            let geom = &self.geometries[info.geometry_idx];

            // Get rotation angle from chromosome
            let rotation_idx = chromosome.rotations.get(instance_idx).copied().unwrap_or(0);
            let rotation_angle = self
                .rotation_angles
                .get(info.geometry_idx)
                .and_then(|angles| angles.get(rotation_idx % angles.len()))
                .copied()
                .unwrap_or(0.0);

            // Compute IFP for this geometry at this rotation
            let ifp = match compute_ifp(&boundary_polygon, geom, rotation_angle) {
                Ok(ifp) => ifp,
                Err(_) => continue,
            };

            if ifp.is_empty() {
                continue;
            }

            // Compute NFPs with all placed geometries
            let mut nfps: Vec<Nfp> = Vec::new();
            for placed in &placed_geometries {
                let placed_exterior = placed.translated_exterior();
                let placed_geom = Geometry2D::new(format!("_placed_{}", placed.geometry.id()))
                    .with_polygon(placed_exterior);

                if let Ok(nfp) = compute_nfp(&placed_geom, geom, rotation_angle) {
                    let expanded = self.expand_nfp(&nfp, spacing);
                    nfps.push(expanded);
                }
            }

            // Shrink IFP by spacing
            let ifp_shrunk = self.shrink_ifp(&ifp, spacing);

            // Find the bottom-left valid placement
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                let placement = Placement::new_2d(
                    geom.id().clone(),
                    info.instance_num,
                    x,
                    y,
                    rotation_angle,
                );

                placements.push(placement);
                placed_geometries.push(PlacedGeometry::new(geom.clone(), (x, y), rotation_angle));
                total_placed_area += geom.measure();
                placed_count += 1;
            }
        }

        let utilization = total_placed_area / self.boundary.measure();
        (placements, utilization, placed_count)
    }

    /// Gets the boundary polygon with margin applied.
    fn get_boundary_polygon_with_margin(&self, margin: f64) -> Vec<(f64, f64)> {
        let (b_min, b_max) = self.boundary.aabb();
        vec![
            (b_min[0] + margin, b_min[1] + margin),
            (b_max[0] - margin, b_min[1] + margin),
            (b_max[0] - margin, b_max[1] - margin),
            (b_min[0] + margin, b_max[1] - margin),
        ]
    }

    /// Computes an adaptive sample step based on geometry sizes.
    fn compute_sample_step(&self) -> f64 {
        if self.geometries.is_empty() {
            return 1.0;
        }

        let mut min_dim = f64::INFINITY;
        for geom in &self.geometries {
            let (g_min, g_max) = geom.aabb();
            let width = g_max[0] - g_min[0];
            let height = g_max[1] - g_min[1];
            min_dim = min_dim.min(width).min(height);
        }

        (min_dim / 4.0).clamp(0.5, 10.0)
    }

    /// Expands an NFP by the given spacing amount.
    fn expand_nfp(&self, nfp: &Nfp, spacing: f64) -> Nfp {
        if spacing <= 0.0 {
            return nfp.clone();
        }

        let expanded_polygons: Vec<Vec<(f64, f64)>> = nfp
            .polygons
            .iter()
            .map(|polygon| {
                let (cx, cy) = polygon_centroid(polygon);
                polygon
                    .iter()
                    .map(|&(x, y)| {
                        let dx = x - cx;
                        let dy = y - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist > 1e-10 {
                            let scale = (dist + spacing) / dist;
                            (cx + dx * scale, cy + dy * scale)
                        } else {
                            (x, y)
                        }
                    })
                    .collect()
            })
            .collect();

        Nfp::from_polygons(expanded_polygons)
    }

    /// Shrinks an IFP by the given spacing amount.
    fn shrink_ifp(&self, ifp: &Nfp, spacing: f64) -> Nfp {
        if spacing <= 0.0 {
            return ifp.clone();
        }

        let shrunk_polygons: Vec<Vec<(f64, f64)>> = ifp
            .polygons
            .iter()
            .filter_map(|polygon| {
                let (cx, cy) = polygon_centroid(polygon);
                let shrunk: Vec<(f64, f64)> = polygon
                    .iter()
                    .map(|&(x, y)| {
                        let dx = x - cx;
                        let dy = y - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist > spacing + 1e-10 {
                            let scale = (dist - spacing) / dist;
                            (cx + dx * scale, cy + dy * scale)
                        } else {
                            (cx, cy)
                        }
                    })
                    .collect();

                if shrunk.len() >= 3 {
                    Some(shrunk)
                } else {
                    None
                }
            })
            .collect();

        Nfp::from_polygons(shrunk_polygons)
    }
}

/// Computes the centroid of a polygon.
fn polygon_centroid(polygon: &[(f64, f64)]) -> (f64, f64) {
    if polygon.is_empty() {
        return (0.0, 0.0);
    }

    let sum: (f64, f64) = polygon.iter().fold((0.0, 0.0), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let n = polygon.len() as f64;
    (sum.0 / n, sum.1 / n)
}

impl GaProblem for NestingProblem {
    type Individual = NestingChromosome;

    fn evaluate(&self, individual: &mut Self::Individual) {
        let (_, utilization, placed_count) = self.decode(individual);

        // Fitness = utilization + bonus for placing all pieces
        let placement_ratio = placed_count as f64 / individual.total_count.max(1) as f64;

        // Primary: maximize placement ratio (most important)
        // Secondary: maximize utilization
        let fitness = placement_ratio * 100.0 + utilization * 10.0;

        individual.set_fitness(fitness, placed_count);
    }

    fn initialize_population<R: Rng>(&self, size: usize, rng: &mut R) -> Vec<Self::Individual> {
        (0..size)
            .map(|_| {
                NestingChromosome::random_with_options(
                    self.num_instances(),
                    self.rotation_options(),
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
            "GA Generation {}: fitness={:.4}, placed={}/{}",
            generation,
            best.fitness(),
            best.placed_count,
            best.total_count
        );
    }
}

/// Runs GA-based nesting optimization.
pub fn run_ga_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    ga_config: GaConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let problem = NestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = GaRunner::new(ga_config, problem);

    // Connect cancellation
    let cancel_handle = runner.cancel_handle();
    let cancelled_clone = cancelled.clone();
    std::thread::spawn(move || {
        while !cancelled_clone.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        cancel_handle.store(true, Ordering::Relaxed);
    });

    let ga_result = runner.run();

    // Decode the best chromosome to get final placements
    let problem = NestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        Arc::new(AtomicBool::new(false)),
    );

    let (placements, utilization, _placed_count) = problem.decode(&ga_result.best);

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
    fn test_nesting_chromosome_crossover() {
        let mut rng = thread_rng();
        let parent1 = NestingChromosome::random_with_options(10, 4, &mut rng);
        let parent2 = NestingChromosome::random_with_options(10, 4, &mut rng);

        let child = parent1.order_crossover(&parent2, &mut rng);

        // Child should be a valid permutation
        assert_eq!(child.order.len(), 10);
        let mut sorted = child.order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_nesting_chromosome_mutation() {
        let mut rng = thread_rng();
        let mut chromosome = NestingChromosome::random_with_options(10, 4, &mut rng);

        chromosome.swap_mutate(&mut rng);

        // Should still be a valid permutation
        let mut sorted = chromosome.order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_ga_nesting_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10);

        let result = run_ga_nesting(
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
    fn test_ga_nesting_all_placed() {
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_ga_nesting(
            &geometries,
            &boundary,
            &config,
            ga_config,
            Arc::new(AtomicBool::new(false)),
        );

        // All 4 pieces should fit easily
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ga_nesting_with_rotation() {
        // L-shaped pieces that might benefit from rotation
        let geometries = vec![
            Geometry2D::rectangle("R1", 30.0, 10.0)
                .with_quantity(3)
                .with_rotations(vec![0.0, 90.0]),
        ];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let ga_config = GaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_ga_nesting(
            &geometries,
            &boundary,
            &config,
            ga_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert!(result.utilization > 0.0);
        // Should be able to place at least some pieces
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_nesting_problem_decode() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = NestingProblem::new(geometries, boundary, config, cancelled);

        assert_eq!(problem.num_instances(), 2);

        // Create a chromosome and decode
        let chromosome = NestingChromosome::new(2, 1);
        let (placements, utilization, placed_count) = problem.decode(&chromosome);

        assert_eq!(placed_count, 2);
        assert_eq!(placements.len(), 2);
        assert!(utilization > 0.0);
    }
}
