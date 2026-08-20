//! Genetic Algorithm based 2D nesting optimization.
//!
//! This module provides GA-based optimization for 2D nesting problems,
//! using the permutation chromosome representation and NFP-guided decoding.

use crate::boundary::Boundary2D;
use crate::clamp_placement_to_boundary;
use crate::geometry::Geometry2D;
use crate::nfp::{
    compute_ifp_with_margin_and_mirror, compute_nfp_mirrored, find_bottom_left_placement,
    verify_no_overlap_mirrored, Nfp, PlacedGeometry,
};
use rand::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::ga::{GaConfig, GaProblem, GaProgress, GaRunner, Individual};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::{Config, ProgressCallback, ProgressInfo};
use u_nesting_core::{Placement, SolveResult};

use crate::placement_utils::{expand_nfp, nesting_fitness, shrink_ifp, InstanceInfo};

/// Nesting chromosome representing a placement order and rotations.
#[derive(Debug, Clone)]
pub struct NestingChromosome {
    /// Permutation of geometry indices (placement order).
    pub order: Vec<usize>,
    /// Rotation index for each geometry instance.
    pub rotations: Vec<usize>,
    /// Mirror flag for each geometry instance (`allow_flip` support).
    ///
    /// Populated unconditionally for every instance, same as `rotations` —
    /// `decode()` masks it against the instance's own `Geometry2D::allow_flip()`,
    /// the same modulo-based tolerance `rotations` already uses for
    /// geometries with fewer rotation options than the population-wide max.
    /// A gene for a non-flippable geometry is simply never read.
    pub mirrors: Vec<bool>,
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
            mirrors: vec![false; num_instances],
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
            .map(|_| rng.random_range(0..rotation_options.max(1)))
            .collect();

        let mirrors: Vec<bool> = (0..num_instances).map(|_| rng.random()).collect();

        Self {
            order,
            rotations,
            mirrors,
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

        // Crossover rotations (uniform)
        let rotations: Vec<usize> = self
            .rotations
            .iter()
            .zip(&other.rotations)
            .map(|(a, b)| if rng.random() { *a } else { *b })
            .collect();

        // Crossover mirrors (uniform), same pattern as rotations.
        let mirrors: Vec<bool> = self
            .mirrors
            .iter()
            .zip(&other.mirrors)
            .map(|(a, b)| if rng.random() { *a } else { *b })
            .collect();

        Self {
            order: child_order,
            rotations,
            mirrors,
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

    /// Rotation mutation.
    pub fn rotation_mutate<R: Rng>(&mut self, rotation_options: usize, rng: &mut R) {
        if self.rotations.is_empty() || rotation_options <= 1 {
            return;
        }

        let idx = rng.random_range(0..self.rotations.len());
        self.rotations[idx] = rng.random_range(0..rotation_options);
        self.fitness = f64::NEG_INFINITY;
    }

    /// Mirror-flag mutation (`allow_flip` support): flips one instance's
    /// mirror bit. Unconditional on the gene, same as `rotation_mutate` —
    /// `decode()` is what masks a flip against the instance's own
    /// `allow_flip()`, so mutating a non-flippable instance's bit is harmless.
    pub fn mirror_mutate<R: Rng>(&mut self, rng: &mut R) {
        if self.mirrors.is_empty() {
            return;
        }

        let idx = rng.random_range(0..self.mirrors.len());
        self.mirrors[idx] = !self.mirrors[idx];
        self.fitness = f64::NEG_INFINITY;
    }

    /// Inversion mutation (reverses a segment).
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
        // 45% swap, 25% inversion, 15% rotation, 15% mirror
        let r: f64 = rng.random();
        if r < 0.45 {
            self.swap_mutate(rng);
        } else if r < 0.70 {
            self.inversion_mutate(rng);
        } else if r < 0.85 {
            // Rotation mutation with 4 options (0, 90, 180, 270 degrees)
            self.rotation_mutate(4, rng);
        } else {
            self.mirror_mutate(rng);
        }
    }
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
            let angles = if angles.is_empty() { vec![0.0] } else { angles };
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
        for &instance_idx in chromosome.order.iter() {
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

            // Mirror flag from chromosome (`allow_flip` support), masked
            // against this instance's own geometry — see `mirrors` doc comment.
            let mirror = chromosome
                .mirrors
                .get(instance_idx)
                .copied()
                .unwrap_or(false)
                && geom.allow_flip();

            // Compute IFP for this geometry at this rotation
            let ifp = match compute_ifp_with_margin_and_mirror(
                &boundary_polygon,
                geom,
                rotation_angle,
                0.0,
                mirror,
            ) {
                Ok(ifp) => ifp,
                Err(_) => {
                    continue;
                }
            };

            if ifp.is_empty() {
                continue;
            }

            // Compute NFPs with all placed geometries
            let mut nfps: Vec<Nfp> = Vec::new();
            for placed in &placed_geometries {
                // Already-mirrored (if applicable) real-world polygon — do
                // NOT mirror it again below, `mirror_stationary=false` always.
                let placed_exterior = placed.translated_exterior();
                let placed_geom = Geometry2D::new(format!("_placed_{}", placed.geometry.id()))
                    .with_polygon(placed_exterior);

                if let Ok(nfp) =
                    compute_nfp_mirrored(&placed_geom, geom, rotation_angle, false, mirror)
                {
                    let expanded = self.expand_nfp(&nfp, spacing);
                    nfps.push(expanded);
                }
            }

            // Shrink IFP by spacing
            let ifp_shrunk = self.shrink_ifp(&ifp, spacing);

            // Find the bottom-left valid placement
            // IFP returns positions where the geometry's origin should be placed.
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            let placement_result = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step);
            if let Some((x, y)) = placement_result {
                // Clamp position to keep geometry within boundary
                // (mirror-aware — an unmirrored AABB has the wrong local
                // extents for a mirrored candidate, see `aabb_at_rotation_mirrored`).
                let geom_aabb = geom.aabb_at_rotation_mirrored(rotation_angle, mirror);
                let boundary_aabb = self.boundary.aabb();

                if let Some((clamped_x, clamped_y)) =
                    clamp_placement_to_boundary(x, y, geom_aabb, boundary_aabb)
                {
                    // Only verify overlap if clamping changed the position
                    // The original NFP-found position is already collision-free by definition
                    let was_clamped = (clamped_x - x).abs() > 1e-6 || (clamped_y - y).abs() > 1e-6;
                    if was_clamped {
                        // Verify no actual polygon overlap using SAT
                        if !verify_no_overlap_mirrored(
                            geom,
                            (clamped_x, clamped_y),
                            rotation_angle,
                            mirror,
                            &placed_geometries,
                        ) {
                            continue; // Skip - clamped position would cause overlap
                        }
                    }

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        info.instance_num,
                        clamped_x,
                        clamped_y,
                        rotation_angle,
                    )
                    .with_mirrored(mirror);

                    placements.push(placement);
                    placed_geometries.push(
                        PlacedGeometry::new(geom.clone(), (clamped_x, clamped_y), rotation_angle)
                            .with_mirrored(mirror),
                    );
                    total_placed_area += geom.measure();
                    placed_count += 1;
                }
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
        expand_nfp(nfp, spacing)
    }

    /// Shrinks an IFP by the given spacing amount.
    fn shrink_ifp(&self, ifp: &Nfp, spacing: f64) -> Nfp {
        shrink_ifp(ifp, spacing)
    }
}

impl GaProblem for NestingProblem {
    type Individual = NestingChromosome;

    fn evaluate(&self, individual: &mut Self::Individual) {
        let (_, utilization, placed_count) = self.decode(individual);
        let fitness = nesting_fitness(placed_count, individual.total_count, utilization);
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

    // Connect cancellation (thread-based polling, not available on WASM)
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

    // Seed the RNG for reproducibility when `config.seed` is set; otherwise use
    // system entropy (non-deterministic).
    let ga_result = match config.seed {
        Some(seed) => runner.run_with_rng(&mut rand::rngs::StdRng::seed_from_u64(seed)),
        None => runner.run(),
    };

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

/// Runs GA-based nesting optimization with progress callback.
pub fn run_ga_nesting_with_progress(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    ga_config: GaConfig,
    cancelled: Arc<AtomicBool>,
    progress_callback: ProgressCallback,
) -> SolveResult<f64> {
    let total_items = geometries.iter().map(|g| g.quantity()).sum::<usize>();

    let problem = NestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = GaRunner::new(ga_config.clone(), problem);

    // Connect cancellation (thread-based polling, not available on WASM)
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

    // Run GA with progress callback adapter. Thread `config.seed` through so the
    // callback-driven path is as reproducible as the plain `run_ga_nesting` path
    // — without this the progress runner fell back to system entropy and a seeded
    // solve was non-deterministic whenever a progress callback was supplied
    // (FFI `solve_2d_with_callback`, the WASM/demo path).
    let max_generations = ga_config.max_generations;
    let progress_adapter = move |ga_progress: GaProgress<f64>| {
        let info = ProgressInfo::new()
            .with_iteration(ga_progress.generation, max_generations)
            .with_fitness(ga_progress.best_fitness)
            .with_utilization(ga_progress.best_fitness) // fitness is utilization
            .with_items(0, total_items) // we don't track placed count during GA
            .with_elapsed(ga_progress.elapsed.as_millis() as u64)
            .with_phase("Genetic Algorithm".to_string());

        let info = if !ga_progress.running {
            info.finished()
        } else {
            info
        };

        progress_callback(info);
    };
    let ga_result = match config.seed {
        Some(seed) => runner.run_with_rng_and_progress(
            &mut rand::rngs::StdRng::seed_from_u64(seed),
            Some(progress_adapter),
        ),
        None => runner.run_with_progress(progress_adapter),
    };

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
        let mut rng = rand::rng();
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
        let mut rng = rand::rng();
        let mut chromosome = NestingChromosome::random_with_options(10, 4, &mut rng);

        chromosome.swap_mutate(&mut rng);

        // Should still be a valid permutation
        let mut sorted = chromosome.order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_nesting_chromosome_mirrors_gene_present() {
        let mut rng = rand::rng();
        let chromosome = NestingChromosome::random_with_options(10, 4, &mut rng);
        assert_eq!(chromosome.mirrors.len(), 10);

        let fixed = NestingChromosome::new(10, 4);
        assert_eq!(fixed.mirrors, vec![false; 10]);
    }

    #[test]
    fn test_nesting_chromosome_mirror_crossover() {
        let mut rng = rand::rng();
        // All-true vs all-false parents — every crossover gene must come
        // from exactly one of the two, so the child is either true or false
        // per position (not some third corrupted value — trivially true for
        // bool, but this also proves the vector is populated per-position,
        // not left at a stale default length).
        let mut parent1 = NestingChromosome::random_with_options(10, 4, &mut rng);
        let mut parent2 = NestingChromosome::random_with_options(10, 4, &mut rng);
        parent1.mirrors = vec![true; 10];
        parent2.mirrors = vec![false; 10];

        let child = parent1.order_crossover(&parent2, &mut rng);
        assert_eq!(child.mirrors.len(), 10);
        assert!(child.mirrors.iter().all(|&m| m || !m)); // always true for bool; length/no-panic is the real assertion
    }

    #[test]
    fn test_mirror_mutate_flips_bit() {
        let mut rng = rand::rng();
        let mut chromosome = NestingChromosome::new(5, 1);
        assert_eq!(chromosome.mirrors, vec![false; 5]);

        // Deterministic single-instance chromosome — flip must change it.
        let mut single = NestingChromosome::new(1, 1);
        single.mirror_mutate(&mut rng);
        assert!(single.mirrors[0]);
        single.mirror_mutate(&mut rng);
        assert!(!single.mirrors[0]);

        chromosome.mirror_mutate(&mut rng);
        assert_eq!(chromosome.mirrors.iter().filter(|&&m| m).count(), 1);
    }

    /// Chiral L-shape — see `nfp.rs`'s `chiral_l` fixture for why this
    /// specific shape (asymmetric width/height/notch, no reflection symmetry).
    fn chiral_l(id: &str) -> Geometry2D {
        Geometry2D::l_shape(id, 30.0, 20.0, 20.0, 10.0)
    }

    fn polygons_overlap(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
        for i in 0..a.len() {
            let (a1, a2) = (a[i], a[(i + 1) % a.len()]);
            for j in 0..b.len() {
                let (b1, b2) = (b[j], b[(j + 1) % b.len()]);
                if crate::polygon_ops::segments_intersect(a1, a2, b1, b2) {
                    return true;
                }
            }
        }
        false
    }

    /// Phase 2 (`allow_flip`/mirroring), GA strategy. Unlike BLF
    /// (`nester.rs`), `NestingProblem::decode()` calls no `.validate()` at
    /// all (only `solve()`'s centralized `validate_geometries` gate does),
    /// so calling it directly exercises the mirror gene deterministically —
    /// useful even now that the public gate is open (Phase 4), since the
    /// GA's own public path is randomized.
    #[test]
    fn test_ga_decode_mirror_no_overlap() {
        let geometries = vec![chiral_l("L").with_flip(true).with_quantity(2)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let config = Config::default().with_spacing(1.0);
        let problem = NestingProblem::new(
            geometries.clone(),
            boundary,
            config,
            Arc::new(AtomicBool::new(false)),
        );

        // Chromosome the GA could produce: place instance 0 first
        // unmirrored, instance 1 second mirrored — decode() applies exactly
        // what the chromosome specifies (the GA's job across generations is
        // to discover which values are good, not decode()'s).
        let mut chromosome = NestingChromosome::new(2, 1);
        chromosome.mirrors = vec![false, true];

        let (placements, utilization, placed_count) = problem.decode(&chromosome);

        assert_eq!(
            placed_count, 2,
            "both instances should fit in this boundary"
        );
        assert_eq!(placements.len(), 2);
        assert!(utilization > 0.0);
        assert!(!placements[0].mirrored);
        assert!(
            placements[1].mirrored,
            "instance 1's mirror gene was true and allow_flip is set — decode() must honor it"
        );

        let poly0 = PlacedGeometry::new(
            geometries[0].clone(),
            (placements[0].x(), placements[0].y()),
            placements[0].angle(),
        )
        .with_mirrored(placements[0].mirrored)
        .translated_exterior();
        let poly1 = PlacedGeometry::new(
            geometries[0].clone(),
            (placements[1].x(), placements[1].y()),
            placements[1].angle(),
        )
        .with_mirrored(placements[1].mirrored)
        .translated_exterior();
        assert!(
            !polygons_overlap(&poly0, &poly1),
            "unmirrored instance 0 and mirrored instance 1 must not overlap"
        );
    }

    #[test]
    fn test_ga_decode_mirror_ignored_without_allow_flip() {
        // allow_flip defaults to false: even a true mirror gene must be
        // masked off by decode() (`&& geom.allow_flip()`), not honored.
        let geometries = vec![chiral_l("L").with_quantity(1)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let problem = NestingProblem::new(
            geometries,
            boundary,
            Config::default(),
            Arc::new(AtomicBool::new(false)),
        );

        let mut chromosome = NestingChromosome::new(1, 1);
        chromosome.mirrors = vec![true];

        let (placements, _utilization, placed_count) = problem.decode(&chromosome);
        assert_eq!(placed_count, 1);
        assert!(
            !placements[0].mirrored,
            "allow_flip=false must suppress the mirror gene"
        );
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
        let geometries = vec![Geometry2D::rectangle("R1", 30.0, 10.0)
            .with_quantity(3)
            .with_rotations(vec![0.0, 90.0])];

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
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2)];

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
