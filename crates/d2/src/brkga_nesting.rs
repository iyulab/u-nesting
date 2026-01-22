//! BRKGA-based 2D nesting optimization.
//!
//! This module provides BRKGA (Biased Random-Key Genetic Algorithm) based
//! optimization for 2D nesting problems. BRKGA uses random-key encoding
//! and biased crossover to favor elite parents.
//!
//! # Random-Key Encoding
//!
//! Each solution is encoded as a vector of random keys in [0, 1):
//! - First N keys: decoded as permutation (placement order)
//! - Next N keys: decoded as rotation indices
//!
//! # Reference
//!
//! Gonçalves, J. F., & Resende, M. G. (2013). A biased random key genetic
//! algorithm for 2D and 3D bin packing problems.

use crate::boundary::Boundary2D;
use crate::clamp_placement_to_boundary;
use crate::geometry::Geometry2D;
use crate::nfp::{compute_ifp, compute_nfp, find_bottom_left_placement, Nfp, PlacedGeometry};
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
}

/// BRKGA problem definition for 2D nesting.
pub struct BrkgaNestingProblem {
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
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl BrkgaNestingProblem {
    /// Creates a new BRKGA nesting problem.
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

        Self {
            geometries,
            boundary,
            config,
            instances,
            rotation_angles,
            cancelled,
        }
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Decodes a chromosome into placements using NFP-guided placement.
    ///
    /// The chromosome keys are interpreted as:
    /// - Keys [0..N): placement order (sorted indices)
    /// - Keys [N..2N): rotation indices (discretized)
    pub fn decode(&self, chromosome: &RandomKeyChromosome) -> (Vec<Placement<f64>>, f64, usize) {
        let n = self.instances.len();
        if n == 0 || chromosome.len() < n {
            return (Vec::new(), 0.0, 0);
        }

        // Decode placement order from first N keys
        let order = chromosome.decode_as_permutation();
        // Only take first N indices (in case chromosome has extra keys)
        let order: Vec<usize> = order.into_iter().take(n).collect();

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

        // Place geometries in the decoded order
        for &instance_idx in &order {
            if self.cancelled.load(Ordering::Relaxed) {
                break;
            }

            if instance_idx >= self.instances.len() {
                continue;
            }

            let info = &self.instances[instance_idx];
            let geom = &self.geometries[info.geometry_idx];

            // Decode rotation from the second half of keys
            let rotation_key_idx = n + instance_idx;
            let num_rotations = self
                .rotation_angles
                .get(info.geometry_idx)
                .map(|a| a.len())
                .unwrap_or(1);

            let rotation_idx = if rotation_key_idx < chromosome.len() {
                chromosome.decode_as_discrete(rotation_key_idx, num_rotations)
            } else {
                0
            };

            let rotation_angle = self
                .rotation_angles
                .get(info.geometry_idx)
                .and_then(|angles| angles.get(rotation_idx))
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
            // IFP returns positions where the geometry's origin should be placed.
            // Clamp to ensure placement keeps geometry within boundary.
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                // Clamp position to keep geometry within boundary
                let geom_aabb = geom.aabb_at_rotation(rotation_angle);
                let boundary_aabb = self.boundary.aabb();

                if let Some((clamped_x, clamped_y)) =
                    clamp_placement_to_boundary(x, y, geom_aabb, boundary_aabb)
                {
                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        info.instance_num,
                        clamped_x,
                        clamped_y,
                        rotation_angle,
                    );

                    placements.push(placement);
                    placed_geometries.push(PlacedGeometry::new(
                        geom.clone(),
                        (clamped_x, clamped_y),
                        rotation_angle,
                    ));
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

    let sum: (f64, f64) = polygon
        .iter()
        .fold((0.0, 0.0), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let n = polygon.len() as f64;
    (sum.0 / n, sum.1 / n)
}

impl BrkgaProblem for BrkgaNestingProblem {
    fn num_keys(&self) -> usize {
        // N keys for order + N keys for rotations
        self.instances.len() * 2
    }

    fn evaluate(&self, chromosome: &mut RandomKeyChromosome) {
        let total_instances = self.instances.len();
        let (_, utilization, placed_count) = self.decode(chromosome);

        // Fitness = utilization + bonus for placing all pieces
        let placement_ratio = placed_count as f64 / total_instances.max(1) as f64;

        // Primary: maximize placement ratio (most important)
        // Secondary: maximize utilization
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
            "BRKGA Generation {}: fitness={:.4}",
            generation,
            best.fitness()
        );
    }
}

/// Runs BRKGA-based nesting optimization.
pub fn run_brkga_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    brkga_config: BrkgaConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let problem = BrkgaNestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled.clone(),
    );

    let runner = BrkgaRunner::with_cancellation(brkga_config, problem, cancelled.clone());

    let brkga_result = runner.run();

    // Decode the best chromosome to get final placements
    let problem = BrkgaNestingProblem::new(
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
    fn test_brkga_nesting_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(20);

        let result = run_brkga_nesting(
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
    fn test_brkga_nesting_all_placed() {
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(30);

        let result = run_brkga_nesting(
            &geometries,
            &boundary,
            &config,
            brkga_config,
            Arc::new(AtomicBool::new(false)),
        );

        // All 4 pieces should fit easily
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_brkga_nesting_with_rotation() {
        let geometries = vec![Geometry2D::rectangle("R1", 30.0, 10.0)
            .with_quantity(3)
            .with_rotations(vec![0.0, 90.0])];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let brkga_config = BrkgaConfig::default()
            .with_population_size(30)
            .with_max_generations(30);

        let result = run_brkga_nesting(
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
        use rand::SeedableRng;

        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2)];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = BrkgaNestingProblem::new(geometries, boundary, config, cancelled);

        assert_eq!(problem.num_instances(), 2);
        // 2 instances * 2 (order + rotation) = 4 keys
        assert_eq!(problem.num_keys(), 4);

        // Create a chromosome with fixed seed for deterministic test
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let chromosome = RandomKeyChromosome::random(problem.num_keys(), &mut rng);
        let (placements, utilization, placed_count) = problem.decode(&chromosome);

        // Decoding should produce valid output (may or may not place items depending on random keys)
        assert_eq!(placements.len(), placed_count);
        if placed_count > 0 {
            assert!(utilization > 0.0);
        }
    }
}
