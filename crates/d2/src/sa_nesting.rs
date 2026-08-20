//! Simulated Annealing-based 2D nesting optimization.
//!
//! This module provides Simulated Annealing based optimization for 2D nesting
//! problems. SA uses neighborhood operators to explore the solution space
//! and accepts worse solutions with a probability that decreases over time.
//!
//! # Neighborhood Operators
//!
//! - **Swap**: Exchange positions of two items in the sequence
//! - **Relocate**: Move an item to a different position
//! - **Inversion**: Reverse a segment of the sequence
//! - **Rotation**: Change the rotation of an item

use crate::boundary::Boundary2D;
use crate::clamp_placement_to_boundary;
use crate::geometry::Geometry2D;
use crate::nfp::{
    compute_ifp_with_margin_and_mirror, compute_nfp_mirrored, find_bottom_left_placement,
    verify_no_overlap_mirrored, Nfp, PlacedGeometry,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::sa::{
    NeighborhoodOperator, PermutationSolution, SaConfig, SaProblem, SaRunner, SaSolution,
};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

use crate::placement_utils::{expand_nfp, nesting_fitness, shrink_ifp, InstanceInfo};

/// SA problem definition for 2D nesting.
pub struct SaNestingProblem {
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
    /// Maximum rotation options across all geometries.
    max_rotation_options: usize,
    /// Whether any geometry allows mirroring (`allow_flip` support) — gates
    /// whether `MirrorFlip` is offered as a neighborhood operator, same
    /// pattern as `max_rotation_options` gating `Rotation`.
    any_allow_flip: bool,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl SaNestingProblem {
    /// Creates a new SA nesting problem.
    pub fn new(
        geometries: Vec<Geometry2D>,
        boundary: Boundary2D,
        config: Config,
        cancelled: Arc<AtomicBool>,
    ) -> Self {
        // Build instance mapping
        let mut instances = Vec::new();
        let mut rotation_angles = Vec::new();
        let mut max_rotation_options = 1;
        let mut any_allow_flip = false;

        for (geom_idx, geom) in geometries.iter().enumerate() {
            // Get rotation angles for this geometry
            let angles = geom.rotations();
            let angles = if angles.is_empty() { vec![0.0] } else { angles };
            max_rotation_options = max_rotation_options.max(angles.len());
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
            max_rotation_options,
            any_allow_flip,
            cancelled,
        }
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Decodes a solution into placements using NFP-guided placement.
    pub fn decode(&self, solution: &PermutationSolution) -> (Vec<Placement<f64>>, f64, usize) {
        let n = self.instances.len();
        if n == 0 || solution.sequence.is_empty() {
            return (Vec::new(), 0.0, 0);
        }

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

        // Place geometries in the solution order
        for (seq_idx, &instance_idx) in solution.sequence.iter().enumerate() {
            if self.cancelled.load(Ordering::Relaxed) {
                break;
            }

            if instance_idx >= self.instances.len() {
                continue;
            }

            let info = &self.instances[instance_idx];
            let geom = &self.geometries[info.geometry_idx];

            // Get rotation from solution
            let rotation_idx = solution.rotations.get(seq_idx).copied().unwrap_or(0);
            let num_rotations = self
                .rotation_angles
                .get(info.geometry_idx)
                .map(|a| a.len())
                .unwrap_or(1);

            let rotation_angle = self
                .rotation_angles
                .get(info.geometry_idx)
                .and_then(|angles| angles.get(rotation_idx % num_rotations))
                .copied()
                .unwrap_or(0.0);

            // Mirror flag from solution (`allow_flip` support), masked
            // against this instance's own geometry — see
            // `PermutationSolution.mirrors` / `NeighborhoodOperator::MirrorFlip`.
            let mirror =
                solution.mirrors.get(seq_idx).copied().unwrap_or(false) && geom.allow_flip();

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
                    let expanded = expand_nfp(&nfp, spacing);
                    nfps.push(expanded);
                }
            }

            // Shrink IFP by spacing
            let ifp_shrunk = shrink_ifp(&ifp, spacing);

            // Find the bottom-left valid placement
            // IFP returns positions where the geometry's origin should be placed.
            // Clamp to ensure placement keeps geometry within boundary.
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
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
}

impl SaProblem for SaNestingProblem {
    type Solution = PermutationSolution;

    fn initial_solution<R: rand::Rng>(&self, rng: &mut R) -> Self::Solution {
        PermutationSolution::random(self.instances.len(), self.max_rotation_options, rng)
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
            NeighborhoodOperator::MirrorFlip => solution.apply_mirror_flip(rng),
        }
    }

    fn evaluate(&self, solution: &mut Self::Solution) {
        let (_, utilization, placed_count) = self.decode(solution);
        let fitness = nesting_fitness(placed_count, self.instances.len(), utilization);
        solution.set_objective(fitness);
    }

    fn available_operators(&self) -> Vec<NeighborhoodOperator> {
        let mut ops = vec![
            NeighborhoodOperator::Swap,
            NeighborhoodOperator::Relocate,
            NeighborhoodOperator::Inversion,
            NeighborhoodOperator::Chain,
        ];
        if self.max_rotation_options > 1 {
            ops.push(NeighborhoodOperator::Rotation);
        }
        if self.any_allow_flip {
            ops.push(NeighborhoodOperator::MirrorFlip);
        }
        ops
    }

    fn on_temperature_change(
        &self,
        temperature: f64,
        iteration: u64,
        best: &Self::Solution,
        _current: &Self::Solution,
    ) {
        log::debug!(
            "SA Iteration {}: temp={:.4}, best_fitness={:.4}",
            iteration,
            temperature,
            best.objective()
        );
    }
}

/// Runs SA-based nesting optimization.
pub fn run_sa_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    sa_config: SaConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let problem = SaNestingProblem::new(
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

    // Seed the RNG for reproducibility when `config.seed` is set; otherwise use
    // system entropy (non-deterministic).
    let sa_result = match config.seed {
        Some(seed) => {
            use rand::SeedableRng;
            runner.run_with_rng(&mut rand::rngs::StdRng::seed_from_u64(seed))
        }
        None => runner.run(),
    };

    // Decode the best solution to get final placements
    let problem = SaNestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        Arc::new(AtomicBool::new(false)),
    );

    let (placements, utilization, _placed_count) = problem.decode(&sa_result.best);

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
    fn test_sa_nesting_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.9)
            .with_iterations_per_temp(20)
            .with_max_iterations(500);

        let result = run_sa_nesting(
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
    fn test_sa_nesting_all_placed() {
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_max_iterations(1000);

        let result = run_sa_nesting(
            &geometries,
            &boundary,
            &config,
            sa_config,
            Arc::new(AtomicBool::new(false)),
        );

        // All 4 pieces should fit easily
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_sa_nesting_with_rotation() {
        let geometries = vec![Geometry2D::rectangle("R1", 30.0, 10.0)
            .with_quantity(3)
            .with_rotations(vec![0.0, 90.0])];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_max_iterations(500);

        let result = run_sa_nesting(
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
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2)];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = SaNestingProblem::new(geometries, boundary, config, cancelled);

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
    fn test_permutation_solution_mirrors_gene_present() {
        let mut rng = rand::rng();
        let solution = PermutationSolution::random(10, 4, &mut rng);
        assert_eq!(solution.mirrors.len(), 10);

        let fixed = PermutationSolution::new(10, 4);
        assert_eq!(fixed.mirrors, vec![false; 10]);
    }

    #[test]
    fn test_apply_mirror_flip_flips_bit() {
        let mut rng = rand::rng();
        let mut single = PermutationSolution::new(1, 1);
        assert!(!single.mirrors[0]);

        single = single.apply_mirror_flip(&mut rng);
        assert!(single.mirrors[0]);
        single = single.apply_mirror_flip(&mut rng);
        assert!(!single.mirrors[0]);
    }

    #[test]
    fn test_available_operators_includes_mirror_flip_only_when_allowed() {
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let cancelled = Arc::new(AtomicBool::new(false));

        let plain = vec![Geometry2D::rectangle("R", 10.0, 10.0).with_quantity(1)];
        let problem = SaNestingProblem::new(
            plain,
            boundary.clone(),
            Config::default(),
            cancelled.clone(),
        );
        assert!(!problem
            .available_operators()
            .contains(&NeighborhoodOperator::MirrorFlip));

        let flippable = vec![Geometry2D::rectangle("R", 10.0, 10.0)
            .with_flip(true)
            .with_quantity(1)];
        let problem = SaNestingProblem::new(flippable, boundary, Config::default(), cancelled);
        assert!(problem
            .available_operators()
            .contains(&NeighborhoodOperator::MirrorFlip));
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

    /// Phase 3 (`allow_flip`/mirroring), SA strategy. Same rationale as
    /// Phase 2's GA test: `SaNestingProblem::decode()` calls no
    /// `.validate()` (only `solve()`'s centralized gate does), so calling it
    /// directly exercises the mirror flag deterministically — useful even
    /// now that the public gate is open (Phase 4), since SA's own public
    /// path is randomized.
    #[test]
    fn test_sa_decode_mirror_no_overlap() {
        let geometries = vec![chiral_l("L").with_flip(true).with_quantity(2)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let config = Config::default().with_spacing(1.0);
        let problem = SaNestingProblem::new(
            geometries.clone(),
            boundary,
            config,
            Arc::new(AtomicBool::new(false)),
        );

        let mut solution = PermutationSolution::new(2, 1);
        solution.mirrors = vec![false, true];

        let (placements, utilization, placed_count) = problem.decode(&solution);

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
    fn test_sa_decode_mirror_ignored_without_allow_flip() {
        let geometries = vec![chiral_l("L").with_quantity(1)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let problem = SaNestingProblem::new(
            geometries,
            boundary,
            Config::default(),
            Arc::new(AtomicBool::new(false)),
        );

        let mut solution = PermutationSolution::new(1, 1);
        solution.mirrors = vec![true];

        let (placements, _utilization, placed_count) = problem.decode(&solution);
        assert_eq!(placed_count, 1);
        assert!(
            !placements[0].mirrored,
            "allow_flip=false must suppress the mirror gene"
        );
    }
}
