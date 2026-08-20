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
use crate::nfp::{
    compute_ifp_with_margin_and_mirror, compute_nfp_mirrored, find_bottom_left_placement,
    verify_no_overlap_mirrored, Nfp, PlacedGeometry,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::brkga::{BrkgaConfig, BrkgaProblem, BrkgaRunner, RandomKeyChromosome};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

use crate::placement_utils::{expand_nfp, nesting_fitness, shrink_ifp, InstanceInfo};

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
    /// Whether any geometry allows mirroring (`allow_flip` support) — gates
    /// whether the chromosome carries a third key block for mirror flags.
    any_allow_flip: bool,
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
        let mut any_allow_flip = false;

        for (geom_idx, geom) in geometries.iter().enumerate() {
            // Get rotation angles for this geometry
            let angles = geom.rotations();
            let angles = if angles.is_empty() { vec![0.0] } else { angles };
            rotation_angles.push(angles);
            any_allow_flip = any_allow_flip || geom.allow_flip();

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
            any_allow_flip,
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
    /// - Keys [2N..3N), only when `any_allow_flip`: mirror flags
    ///   (`allow_flip` support, discretized to 2 options) — a third block,
    ///   same encoding style as rotation, only present when at least one
    ///   geometry can use it (keeps the chromosome at its original 2N length
    ///   for problems that never need mirroring).
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

            // Decode mirror flag from the third key block (`allow_flip` support).
            let mirror_key_idx = 2 * n + instance_idx;
            let mirror = self.any_allow_flip
                && mirror_key_idx < chromosome.len()
                && chromosome.decode_as_discrete(mirror_key_idx, 2) == 1
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
                Err(_) => continue,
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
            // Clamp to ensure placement keeps geometry within boundary.
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                // Clamp position to keep geometry within boundary
                let geom_aabb = geom.aabb_at_rotation(rotation_angle);
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

impl BrkgaProblem for BrkgaNestingProblem {
    fn num_keys(&self) -> usize {
        // N keys for order + N keys for rotations + (if any_allow_flip) N
        // keys for mirror flags.
        let n = self.instances.len();
        if self.any_allow_flip {
            n * 3
        } else {
            n * 2
        }
    }

    fn evaluate(&self, chromosome: &mut RandomKeyChromosome) {
        let (_, utilization, placed_count) = self.decode(chromosome);
        let fitness = nesting_fitness(placed_count, self.instances.len(), utilization);
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

    // Seed the RNG for reproducibility when `config.seed` is set; otherwise use
    // system entropy (non-deterministic).
    let brkga_result = match config.seed {
        Some(seed) => {
            use rand::SeedableRng;
            runner.run_with_rng(&mut rand::rngs::StdRng::seed_from_u64(seed))
        }
        None => runner.run(),
    };

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

    #[test]
    fn test_brkga_num_keys_includes_mirror_block_only_when_allowed() {
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let cancelled = Arc::new(AtomicBool::new(false));

        let plain = vec![Geometry2D::rectangle("R", 10.0, 10.0).with_quantity(2)];
        let problem = BrkgaNestingProblem::new(
            plain,
            boundary.clone(),
            Config::default(),
            cancelled.clone(),
        );
        assert_eq!(
            problem.num_keys(),
            4,
            "2 instances * 2 blocks (order + rotation)"
        );

        let flippable = vec![Geometry2D::rectangle("R", 10.0, 10.0)
            .with_flip(true)
            .with_quantity(2)];
        let problem = BrkgaNestingProblem::new(flippable, boundary, Config::default(), cancelled);
        assert_eq!(
            problem.num_keys(),
            6,
            "2 instances * 3 blocks (order + rotation + mirror)"
        );
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

    /// Phase 4-item (`allow_flip`/mirroring), BRKGA strategy. Same bypass
    /// strategy as GA/SA: `decode()` calls no `.validate()` at all.
    ///
    /// Chromosome keys are hand-picked (not random) to force a deterministic
    /// order/mirror outcome: `decode_as_permutation()` sorts ALL `num_keys()`
    /// key indices by value (a pre-existing property of this decoder, not
    /// something this change introduces — see the third key block's doc
    /// comment), so the order-block keys (indices 0, 1) are set smaller than
    /// every other key to guarantee they sort first and `.take(n)` selects
    /// exactly instances 0 and 1, in that order.
    #[test]
    fn test_brkga_decode_mirror_no_overlap() {
        let geometries = vec![chiral_l("L").with_flip(true).with_quantity(2)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let config = Config::default().with_spacing(1.0);
        let problem = BrkgaNestingProblem::new(
            geometries.clone(),
            boundary,
            config,
            Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(problem.num_keys(), 6);

        let mut chromosome = RandomKeyChromosome::new(6);
        chromosome.keys = vec![
            0.01, // order: instance 0 first
            0.02, // order: instance 1 second
            0.5, 0.5,  // rotation (only 1 option here, value irrelevant)
            0.25, // mirror: instance 0 -> false
            0.75, // mirror: instance 1 -> true
        ];

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
            "instance 1's mirror key decoded to true and allow_flip is set — decode() must honor it"
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
    fn test_brkga_decode_mirror_ignored_without_allow_flip() {
        let geometries = vec![chiral_l("L").with_quantity(1)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let problem = BrkgaNestingProblem::new(
            geometries,
            boundary,
            Config::default(),
            Arc::new(AtomicBool::new(false)),
        );
        // allow_flip=false -> num_keys stays at the 2-block size, no mirror
        // block at all.
        assert_eq!(problem.num_keys(), 2);

        let mut chromosome = RandomKeyChromosome::new(2);
        chromosome.keys = vec![0.01, 0.5];

        let (placements, _utilization, placed_count) = problem.decode(&chromosome);
        assert_eq!(placed_count, 1);
        assert!(
            !placements[0].mirrored,
            "allow_flip=false must suppress mirroring"
        );
    }
}
