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
use crate::geometry::Geometry2D;
use crate::nfp::{compute_ifp, compute_nfp, find_bottom_left_placement, Nfp, PlacedGeometry};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::sa::{
    NeighborhoodOperator, PermutationSolution, SaConfig, SaProblem, SaRunner, SaSolution,
};
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

        for (geom_idx, geom) in geometries.iter().enumerate() {
            // Get rotation angles for this geometry
            let angles = geom.rotations();
            let angles = if angles.is_empty() { vec![0.0] } else { angles };
            max_rotation_options = max_rotation_options.max(angles.len());
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
            max_rotation_options,
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
                    let expanded = expand_nfp(&nfp, spacing);
                    nfps.push(expanded);
                }
            }

            // Shrink IFP by spacing
            let ifp_shrunk = shrink_ifp(&ifp, spacing);

            // Find the bottom-left valid placement
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                // Adjust for geometry's own coordinate offset
                let (g_min, _) = geom.aabb_at_rotation(rotation_angle);
                let adjusted_x = x - g_min[0];
                let adjusted_y = y - g_min[1];

                let placement =
                    Placement::new_2d(geom.id().clone(), info.instance_num, adjusted_x, adjusted_y, rotation_angle);

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
        }
    }

    fn evaluate(&self, solution: &mut Self::Solution) {
        let total_instances = self.instances.len();
        let (_, utilization, placed_count) = self.decode(solution);

        // Fitness = utilization + bonus for placing all pieces
        let placement_ratio = placed_count as f64 / total_instances.max(1) as f64;

        // Primary: maximize placement ratio (most important)
        // Secondary: maximize utilization
        let fitness = placement_ratio * 100.0 + utilization * 10.0;

        solution.set_objective(fitness);
    }

    fn available_operators(&self) -> Vec<NeighborhoodOperator> {
        if self.max_rotation_options > 1 {
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
            "SA Iteration {}: temp={:.4}, best_fitness={:.4}",
            iteration,
            temperature,
            best.objective()
        );
    }
}

/// Expands an NFP by the given spacing amount.
fn expand_nfp(nfp: &Nfp, spacing: f64) -> Nfp {
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
fn shrink_ifp(ifp: &Nfp, spacing: f64) -> Nfp {
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

    // Set cancellation
    let cancel_handle = runner.cancel_handle();
    let cancelled_clone = cancelled.clone();
    std::thread::spawn(move || {
        while !cancelled_clone.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        cancel_handle.store(true, Ordering::Relaxed);
    });

    let sa_result = runner.run();

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
        use rand::prelude::*;

        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2)];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = SaNestingProblem::new(geometries, boundary, config, cancelled);

        assert_eq!(problem.num_instances(), 2);

        // Create a random solution and decode
        let mut rng = thread_rng();
        let solution = PermutationSolution::random(problem.num_instances(), 1, &mut rng);
        let (placements, utilization, placed_count) = problem.decode(&solution);

        // Should place at least one item
        assert!(placed_count >= 1);
        assert_eq!(placements.len(), placed_count);
        if placed_count > 0 {
            assert!(utilization > 0.0);
        }
    }
}
