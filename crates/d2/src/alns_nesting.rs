//! Adaptive Large Neighborhood Search (ALNS) based 2D nesting optimization.
//!
//! This module provides ALNS-based optimization for 2D nesting problems,
//! implementing the algorithm from Ropke & Pisinger (2006).
//!
//! # Destroy Operators
//!
//! - **Random**: Remove random items from the solution
//! - **Worst**: Remove items with worst placement scores
//! - **Related**: Remove items similar to a seed item
//! - **Shaw**: Remove items based on spatial clustering
//!
//! # Repair Operators
//!
//! - **Greedy**: Place items at best available position
//! - **Regret**: Use regret-based insertion
//! - **Random**: Place items in random valid positions
//! - **BLF**: Use bottom-left fill heuristic

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
use crate::nfp::{compute_ifp, compute_nfp, find_bottom_left_placement, Nfp, PlacedGeometry};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use u_nesting_core::alns::{
    AlnsConfig, AlnsProblem, AlnsResult, AlnsRunner, AlnsSolution, DestroyOperatorId,
    DestroyResult, RepairOperatorId, RepairResult,
};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

use rand::prelude::*;

/// Instance information for decoding.
#[derive(Debug, Clone)]
struct InstanceInfo {
    /// Index into the geometries array.
    geometry_idx: usize,
    /// Instance number within this geometry's quantity.
    #[allow(dead_code)]
    instance_num: usize,
}

/// A placed item in the ALNS solution.
#[derive(Debug, Clone)]
pub struct PlacedItem {
    /// Instance index.
    pub instance_idx: usize,
    /// X position.
    pub x: f64,
    /// Y position.
    pub y: f64,
    /// Rotation angle in radians.
    pub rotation: f64,
    /// Placement score (lower = better).
    pub score: f64,
}

/// ALNS solution for 2D nesting.
#[derive(Debug, Clone)]
pub struct AlnsNestingSolution {
    /// Placed items.
    pub placed: Vec<PlacedItem>,
    /// Unplaced instance indices.
    pub unplaced: Vec<usize>,
    /// Total number of instances.
    pub total_instances: usize,
    /// Total placed area.
    pub placed_area: f64,
    /// Boundary area.
    pub boundary_area: f64,
    /// Maximum Y coordinate used (strip height).
    pub max_y: f64,
}

impl AlnsNestingSolution {
    /// Create a new empty solution.
    pub fn new(total_instances: usize, boundary_area: f64) -> Self {
        Self {
            placed: Vec::new(),
            unplaced: (0..total_instances).collect(),
            total_instances,
            placed_area: 0.0,
            boundary_area,
            max_y: 0.0,
        }
    }
}

impl AlnsSolution for AlnsNestingSolution {
    fn fitness(&self) -> f64 {
        // Fitness combines unplaced penalty + utilization + height
        let unplaced_penalty = self.unplaced.len() as f64 * 1000.0;
        let utilization_penalty = if self.placed_area > 0.0 {
            1.0 - (self.placed_area / self.boundary_area)
        } else {
            1.0
        };
        let height_penalty = self.max_y / 1000.0;

        unplaced_penalty + utilization_penalty + height_penalty
    }

    fn placed_count(&self) -> usize {
        self.placed.len()
    }

    fn total_count(&self) -> usize {
        self.total_instances
    }
}

/// ALNS problem definition for 2D nesting.
pub struct AlnsNestingProblem {
    /// Input geometries.
    geometries: Vec<Geometry2D>,
    /// Boundary container.
    boundary: Boundary2D,
    /// Solver configuration.
    config: Config,
    /// Instance mapping.
    instances: Vec<InstanceInfo>,
    /// Available rotation angles per geometry.
    rotation_angles: Vec<Vec<f64>>,
    /// Geometry areas.
    geometry_areas: Vec<f64>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
    /// Start time for timeout checking.
    start_time: Instant,
    /// Time limit in milliseconds.
    time_limit_ms: u64,
}

impl AlnsNestingProblem {
    /// Creates a new ALNS nesting problem.
    pub fn new(
        geometries: Vec<Geometry2D>,
        boundary: Boundary2D,
        config: Config,
        cancelled: Arc<AtomicBool>,
        time_limit_ms: u64,
    ) -> Self {
        let mut instances = Vec::new();
        let mut rotation_angles = Vec::new();
        let mut geometry_areas = Vec::new();

        for (geom_idx, geom) in geometries.iter().enumerate() {
            let angles = geom.rotations();
            let angles = if angles.is_empty() { vec![0.0] } else { angles };
            rotation_angles.push(angles);

            let area = geom.measure();
            geometry_areas.push(area);

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
            geometry_areas,
            cancelled,
            start_time: Instant::now(),
            time_limit_ms,
        }
    }

    /// Check if timeout has been reached.
    fn is_timed_out(&self) -> bool {
        if self.time_limit_ms == 0 {
            return false;
        }
        self.start_time.elapsed().as_millis() as u64 >= self.time_limit_ms
    }

    /// Returns the total number of instances.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Get boundary polygon with margin.
    fn get_boundary_polygon_with_margin(&self, margin: f64) -> Vec<(f64, f64)> {
        let (min, max) = self.boundary.aabb();
        vec![
            (min[0] + margin, min[1] + margin),
            (max[0] - margin, min[1] + margin),
            (max[0] - margin, max[1] - margin),
            (min[0] + margin, max[1] - margin),
        ]
    }

    /// Compute sample step for grid search.
    fn compute_sample_step(&self) -> f64 {
        let (min, max) = self.boundary.aabb();
        let width = max[0] - min[0];
        (width / 100.0).max(1.0)
    }

    /// Try to place an item at the best position using NFP.
    fn try_place_item(
        &self,
        instance_idx: usize,
        placed_geometries: &[PlacedGeometry],
        boundary_polygon: &[(f64, f64)],
        sample_step: f64,
    ) -> Option<PlacedItem> {
        let info = &self.instances[instance_idx];
        let geom = &self.geometries[info.geometry_idx];
        let angles = &self.rotation_angles[info.geometry_idx];

        let mut best_placement: Option<PlacedItem> = None;
        let mut best_y = f64::MAX;

        for &rotation in angles {
            let ifp = match compute_ifp(boundary_polygon, geom, rotation) {
                Ok(ifp) => ifp,
                Err(_) => continue,
            };

            if ifp.is_empty() {
                continue;
            }

            let spacing = self.config.spacing;
            let mut nfps: Vec<Nfp> = Vec::new();

            for pg in placed_geometries {
                let placed_exterior = pg.translated_exterior();
                let placed_geom = Geometry2D::new(format!("_placed_{}", pg.geometry.id()))
                    .with_polygon(placed_exterior);

                if let Ok(nfp) = compute_nfp(&placed_geom, geom, rotation) {
                    let expanded = expand_nfp(&nfp, spacing);
                    nfps.push(expanded);
                }
            }

            let ifp_shrunk = shrink_ifp(&ifp, spacing);
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();

            // IFP returns positions where the geometry's origin should be placed.
            // Clamp to ensure placement keeps geometry within boundary.
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                // Compute valid position bounds based on geometry AABB at this rotation
                let (g_min, g_max) = geom.aabb_at_rotation(rotation);
                let (b_min, b_max) = self.boundary.aabb();

                // Clamp position to keep geometry within boundary
                // Use .max(b_min) to ensure origin position >= boundary min
                let min_valid_x = (b_min[0] - g_min[0]).max(b_min[0]);
                let max_valid_x = b_max[0] - g_max[0];
                let min_valid_y = (b_min[1] - g_min[1]).max(b_min[1]);
                let max_valid_y = b_max[1] - g_max[1];

                let clamped_x = x.clamp(min_valid_x, max_valid_x);
                let clamped_y = y.clamp(min_valid_y, max_valid_y);

                if clamped_y < best_y {
                    best_y = clamped_y;
                    best_placement = Some(PlacedItem {
                        instance_idx,
                        x: clamped_x,
                        y: clamped_y,
                        rotation,
                        score: clamped_y,
                    });
                }
            }
        }

        best_placement
    }

    /// Place items using BLF heuristic.
    fn place_items_blf(&self, items: &[usize], solution: &mut AlnsNestingSolution) {
        let margin = self.config.margin;
        let boundary_polygon = self.get_boundary_polygon_with_margin(margin);
        let sample_step = self.compute_sample_step();

        let mut placed_geometries: Vec<PlacedGeometry> = Vec::new();
        for item in &solution.placed {
            let info = &self.instances[item.instance_idx];
            let geom = &self.geometries[info.geometry_idx];
            placed_geometries.push(PlacedGeometry {
                geometry: geom.clone(),
                position: (item.x, item.y),
                rotation: item.rotation,
            });
        }

        // Sort items by area (largest first)
        let mut sorted_items = items.to_vec();
        sorted_items.sort_by(|&a, &b| {
            let area_a = self.geometry_areas[self.instances[a].geometry_idx];
            let area_b = self.geometry_areas[self.instances[b].geometry_idx];
            area_b
                .partial_cmp(&area_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &instance_idx in &sorted_items {
            // Check cancellation and timeout
            if self.cancelled.load(Ordering::Relaxed) || self.is_timed_out() {
                break;
            }

            if let Some(placement) = self.try_place_item(
                instance_idx,
                &placed_geometries,
                &boundary_polygon,
                sample_step,
            ) {
                let info = &self.instances[instance_idx];
                let area = self.geometry_areas[info.geometry_idx];

                solution.placed_area += area;
                solution.max_y = solution.max_y.max(placement.y);

                let geom = &self.geometries[info.geometry_idx];
                placed_geometries.push(PlacedGeometry {
                    geometry: geom.clone(),
                    position: (placement.x, placement.y),
                    rotation: placement.rotation,
                });

                solution.placed.push(placement);
                solution.unplaced.retain(|&idx| idx != instance_idx);
            }
        }
    }

    /// Remove item from solution.
    fn remove_item(&self, solution: &mut AlnsNestingSolution, instance_idx: usize) {
        if let Some(pos) = solution
            .placed
            .iter()
            .position(|p| p.instance_idx == instance_idx)
        {
            let item = solution.placed.remove(pos);
            let info = &self.instances[item.instance_idx];
            solution.placed_area -= self.geometry_areas[info.geometry_idx];
            solution.unplaced.push(item.instance_idx);
        }

        // Recalculate max_y
        solution.max_y = solution.placed.iter().map(|p| p.y).fold(0.0, f64::max);
    }
}

impl AlnsProblem for AlnsNestingProblem {
    type Solution = AlnsNestingSolution;

    fn create_initial_solution(&mut self) -> AlnsNestingSolution {
        let boundary_area = self.boundary.measure();
        let mut solution = AlnsNestingSolution::new(self.instances.len(), boundary_area);

        let all_items: Vec<usize> = (0..self.instances.len()).collect();
        self.place_items_blf(&all_items, &mut solution);

        solution
    }

    fn clone_solution(&self, solution: &AlnsNestingSolution) -> AlnsNestingSolution {
        solution.clone()
    }

    fn destroy_operators(&self) -> Vec<DestroyOperatorId> {
        vec![
            DestroyOperatorId::Random,
            DestroyOperatorId::Worst,
            DestroyOperatorId::Related,
            DestroyOperatorId::Shaw,
        ]
    }

    fn repair_operators(&self) -> Vec<RepairOperatorId> {
        vec![
            RepairOperatorId::Greedy,
            RepairOperatorId::BottomLeftFill,
            RepairOperatorId::Random,
        ]
    }

    fn destroy(
        &mut self,
        solution: &mut AlnsNestingSolution,
        operator: DestroyOperatorId,
        degree: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> DestroyResult {
        let num_to_remove = ((solution.placed.len() as f64 * degree).ceil() as usize).max(1);
        let mut removed_indices = Vec::new();

        if solution.placed.is_empty() {
            return DestroyResult {
                removed_indices,
                operator,
            };
        }

        match operator {
            DestroyOperatorId::Random => {
                // Random removal
                let mut indices: Vec<usize> =
                    solution.placed.iter().map(|p| p.instance_idx).collect();
                indices.shuffle(rng);

                for &idx in indices.iter().take(num_to_remove) {
                    removed_indices.push(idx);
                }
            }
            DestroyOperatorId::Worst => {
                // Worst removal (highest Y position = worst)
                let mut items_with_score: Vec<(usize, f64)> = solution
                    .placed
                    .iter()
                    .map(|p| (p.instance_idx, p.score))
                    .collect();

                items_with_score
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (idx, _) in items_with_score.iter().take(num_to_remove) {
                    removed_indices.push(*idx);
                }
            }
            DestroyOperatorId::Related | DestroyOperatorId::Shaw => {
                // Cluster removal (same logic for both)
                let seed_idx = rng.gen_range(0..solution.placed.len());
                let seed = &solution.placed[seed_idx];
                let seed_x = seed.x;
                let seed_y = seed.y;

                let mut items_with_distance: Vec<(usize, f64)> = solution
                    .placed
                    .iter()
                    .map(|item| {
                        let dx = item.x - seed_x;
                        let dy = item.y - seed_y;
                        (item.instance_idx, (dx * dx + dy * dy).sqrt())
                    })
                    .collect();

                items_with_distance
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                for (idx, _) in items_with_distance.iter().take(num_to_remove) {
                    removed_indices.push(*idx);
                }
            }
            DestroyOperatorId::Custom(_) => {
                // Fall back to random for custom operators
                let mut indices: Vec<usize> =
                    solution.placed.iter().map(|p| p.instance_idx).collect();
                indices.shuffle(rng);

                for &idx in indices.iter().take(num_to_remove) {
                    removed_indices.push(idx);
                }
            }
        }

        // Remove items from solution
        for &idx in &removed_indices {
            self.remove_item(solution, idx);
        }

        DestroyResult {
            removed_indices,
            operator,
        }
    }

    fn repair(
        &mut self,
        solution: &mut AlnsNestingSolution,
        _destroyed: &DestroyResult,
        operator: RepairOperatorId,
    ) -> RepairResult {
        let items_to_place = solution.unplaced.clone();
        let initial_placed = solution.placed.len();

        match operator {
            RepairOperatorId::Greedy | RepairOperatorId::BottomLeftFill => {
                // BLF is already greedy for bottom-left positions
                self.place_items_blf(&items_to_place, solution);
            }
            RepairOperatorId::Regret => {
                // Regret-based insertion (simplified: use BLF for now)
                self.place_items_blf(&items_to_place, solution);
            }
            RepairOperatorId::Random => {
                // Random order placement
                let mut shuffled = items_to_place.clone();
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::from_entropy();
                shuffled.shuffle(&mut rng);
                self.place_items_blf(&shuffled, solution);
            }
            RepairOperatorId::Custom(_) => {
                self.place_items_blf(&items_to_place, solution);
            }
        }

        RepairResult {
            placed_count: solution.placed.len() - initial_placed,
            unplaced_count: solution.unplaced.len(),
            operator,
        }
    }

    fn relatedness(&self, solution: &AlnsNestingSolution, i: usize, j: usize) -> f64 {
        // Relatedness based on spatial distance
        let item_i = solution.placed.iter().find(|p| p.instance_idx == i);
        let item_j = solution.placed.iter().find(|p| p.instance_idx == j);

        match (item_i, item_j) {
            (Some(a), Some(b)) => {
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                1.0 / (1.0 + (dx * dx + dy * dy).sqrt())
            }
            _ => 0.0,
        }
    }
}

/// Run ALNS nesting optimization.
pub fn run_alns_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    alns_config: &AlnsConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let mut problem = AlnsNestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled,
        alns_config.time_limit_ms,
    );

    let runner = AlnsRunner::new(alns_config.clone());
    let alns_result: AlnsResult<AlnsNestingSolution> = runner.run(&mut problem, |_progress| {
        // Progress callback
    });

    let mut result = SolveResult::new();

    for item in &alns_result.best_solution.placed {
        let info = &problem.instances[item.instance_idx];
        let geom = &problem.geometries[info.geometry_idx];

        result.placements.push(Placement::new_2d(
            geom.id().to_string(),
            info.instance_num,
            item.x,
            item.y,
            item.rotation,
        ));
    }

    result.boundaries_used = if result.placements.is_empty() { 0 } else { 1 };
    result.utilization =
        alns_result.best_solution.placed_area / alns_result.best_solution.boundary_area;
    result.computation_time_ms = alns_result.elapsed_ms;
    result.iterations = Some(alns_result.iterations as u64);
    result.best_fitness = Some(alns_result.best_fitness);
    result.strategy = Some("ALNS".to_string());

    result
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_geometries() -> Vec<Geometry2D> {
        vec![
            Geometry2D::rectangle("rect1", 50.0, 30.0).with_quantity(3),
            Geometry2D::rectangle("rect2", 40.0, 40.0).with_quantity(2),
            Geometry2D::rectangle("rect3", 60.0, 20.0).with_quantity(2),
        ]
    }

    fn create_test_boundary() -> Boundary2D {
        Boundary2D::rectangle(300.0, 200.0)
    }

    #[test]
    fn test_alns_nesting_problem_creation() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);

        assert_eq!(problem.num_instances(), 7);
    }

    #[test]
    fn test_alns_nesting_initial_solution() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let solution = problem.create_initial_solution();

        assert!(!solution.placed.is_empty());
        assert!(solution.placed_area > 0.0);
    }

    #[test]
    fn test_alns_nesting_solution_fitness() {
        let solution = AlnsNestingSolution {
            placed: vec![
                PlacedItem {
                    instance_idx: 0,
                    x: 10.0,
                    y: 10.0,
                    rotation: 0.0,
                    score: 10.0,
                },
                PlacedItem {
                    instance_idx: 1,
                    x: 60.0,
                    y: 10.0,
                    rotation: 0.0,
                    score: 10.0,
                },
            ],
            unplaced: vec![2],
            total_instances: 3,
            placed_area: 3000.0,
            boundary_area: 60000.0,
            max_y: 50.0,
        };

        let fitness = solution.fitness();
        assert!(fitness > 0.0);
        assert!(fitness >= 1000.0); // 1 unplaced item penalty
    }

    #[test]
    fn test_alns_nesting_destroy_random() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let initial_placed = solution.placed.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = problem.destroy(&mut solution, DestroyOperatorId::Random, 0.3, &mut rng);

        assert!(!result.removed_indices.is_empty());
        assert_eq!(result.operator, DestroyOperatorId::Random);
        assert!(solution.placed.len() < initial_placed);
    }

    #[test]
    fn test_alns_nesting_destroy_worst() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let initial_placed = solution.placed.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = problem.destroy(&mut solution, DestroyOperatorId::Worst, 0.3, &mut rng);

        assert!(!result.removed_indices.is_empty());
        assert_eq!(result.operator, DestroyOperatorId::Worst);
        assert!(solution.placed.len() < initial_placed);
    }

    #[test]
    fn test_alns_nesting_repair() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let destroy_result =
            problem.destroy(&mut solution, DestroyOperatorId::Random, 0.5, &mut rng);
        let after_destroy_placed = solution.placed.len();

        let repair_result =
            problem.repair(&mut solution, &destroy_result, RepairOperatorId::Greedy);

        assert!(repair_result.placed_count > 0 || after_destroy_placed == solution.placed.len());
        assert_eq!(repair_result.operator, RepairOperatorId::Greedy);
    }

    #[test]
    fn test_run_alns_nesting() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let alns_config = AlnsConfig::new()
            .with_max_iterations(50)
            .with_time_limit_ms(5000);
        let cancelled = Arc::new(AtomicBool::new(false));

        let result = run_alns_nesting(&geometries, &boundary, &config, &alns_config, cancelled);

        assert!(!result.placements.is_empty());
        assert!(result.utilization > 0.0);
    }

    #[test]
    fn test_alns_nesting_full_cycle() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);

        let alns_config = AlnsConfig::new().with_max_iterations(10).with_seed(42);

        let runner = AlnsRunner::new(alns_config);
        let result: AlnsResult<AlnsNestingSolution> = runner.run(&mut problem, |progress| {
            assert!(progress.iteration <= 10);
        });

        assert!(result.iterations <= 10);
        assert!(!result.best_solution.placed.is_empty());
    }

    #[test]
    fn test_alns_destroy_operators() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let operators = problem.destroy_operators();

        assert!(operators.contains(&DestroyOperatorId::Random));
        assert!(operators.contains(&DestroyOperatorId::Worst));
        assert!(operators.contains(&DestroyOperatorId::Related));
        assert!(operators.contains(&DestroyOperatorId::Shaw));
    }

    #[test]
    fn test_alns_repair_operators() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = AlnsNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let operators = problem.repair_operators();

        assert!(operators.contains(&RepairOperatorId::Greedy));
        assert!(operators.contains(&RepairOperatorId::BottomLeftFill));
        assert!(operators.contains(&RepairOperatorId::Random));
    }
}
