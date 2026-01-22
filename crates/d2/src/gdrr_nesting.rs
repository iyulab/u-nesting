//! Goal-Driven Ruin and Recreate (GDRR) based 2D nesting optimization.
//!
//! This module provides GDRR-based optimization for 2D nesting problems,
//! implementing the algorithm from Gardeyn & Wauters (EJOR 2022).
//!
//! # Ruin Operators
//!
//! - **Random**: Remove random items from the solution
//! - **Cluster**: Remove spatially clustered items
//! - **Worst**: Remove items with worst placement scores
//!
//! # Recreate Operators
//!
//! - **BestFit**: Place items using best-fit decreasing by area
//! - **BLF**: Use bottom-left fill heuristic
//! - **NFP**: NFP-guided placement for optimal positioning

use crate::boundary::Boundary2D;
use crate::clamp_placement_to_boundary;
use crate::geometry::Geometry2D;
use crate::nfp::{compute_ifp, compute_nfp, find_bottom_left_placement, Nfp, PlacedGeometry};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use u_nesting_core::gdrr::{
    GdrrConfig, GdrrProblem, GdrrResult, GdrrRunner, GdrrSolution, RecreateResult, RecreateType,
    RuinResult, RuinType, RuinedItem,
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

/// A placed item in the GDRR solution.
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
    /// Placement score (lower = better, based on position).
    pub score: f64,
}

/// GDRR solution for 2D nesting.
#[derive(Debug, Clone)]
pub struct GdrrNestingSolution {
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

impl GdrrNestingSolution {
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

impl GdrrSolution for GdrrNestingSolution {
    fn fitness(&self) -> f64 {
        // Fitness combines:
        // 1. Penalty for unplaced items (high weight)
        // 2. Inverse of utilization (lower is better)
        // 3. Strip height minimization
        let unplaced_penalty = self.unplaced.len() as f64 * 1000.0;
        let utilization_penalty = if self.placed_area > 0.0 {
            1.0 - (self.placed_area / self.boundary_area)
        } else {
            1.0
        };
        let height_penalty = self.max_y / 1000.0; // Normalized height

        unplaced_penalty + utilization_penalty + height_penalty
    }

    fn placed_count(&self) -> usize {
        self.placed.len()
    }

    fn total_count(&self) -> usize {
        self.total_instances
    }

    fn utilization(&self) -> f64 {
        if self.boundary_area > 0.0 {
            self.placed_area / self.boundary_area
        } else {
            0.0
        }
    }

    fn fits_goal(&self, goal: f64) -> bool {
        // Goal represents target utilization or strip height
        self.fitness() <= goal
    }
}

/// GDRR problem definition for 2D nesting.
pub struct GdrrNestingProblem {
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

impl GdrrNestingProblem {
    /// Creates a new GDRR nesting problem.
    pub fn new(
        geometries: Vec<Geometry2D>,
        boundary: Boundary2D,
        config: Config,
        cancelled: Arc<AtomicBool>,
        time_limit_ms: u64,
    ) -> Self {
        // Build instance mapping
        let mut instances = Vec::new();
        let mut rotation_angles = Vec::new();
        let mut geometry_areas = Vec::new();

        for (geom_idx, geom) in geometries.iter().enumerate() {
            // Get rotation angles
            let angles = geom.rotations();
            let angles = if angles.is_empty() { vec![0.0] } else { angles };
            rotation_angles.push(angles);

            // Compute area
            let area = geom.measure();
            geometry_areas.push(area);

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
            // Compute IFP
            let ifp = match compute_ifp(boundary_polygon, geom, rotation) {
                Ok(ifp) => ifp,
                Err(_) => continue,
            };

            if ifp.is_empty() {
                continue;
            }

            // Compute NFPs with placed geometries
            let spacing = self.config.spacing;
            let mut nfps: Vec<Nfp> = Vec::new();

            for pg in placed_geometries {
                // Create temporary geometry with translated exterior
                let placed_exterior = pg.translated_exterior();
                let placed_geom = Geometry2D::new(format!("_placed_{}", pg.geometry.id()))
                    .with_polygon(placed_exterior);

                if let Ok(nfp) = compute_nfp(&placed_geom, geom, rotation) {
                    let expanded = expand_nfp(&nfp, spacing);
                    nfps.push(expanded);
                }
            }

            // Shrink IFP by spacing
            let ifp_shrunk = shrink_ifp(&ifp, spacing);

            // Find bottom-left placement
            // IFP returns positions where the geometry's origin should be placed.
            // Clamp to ensure placement keeps geometry within boundary.
            let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
            if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                // Clamp position to keep geometry within boundary
                let geom_aabb = geom.aabb_at_rotation(rotation);
                let boundary_aabb = self.boundary.aabb();

                if let Some((clamped_x, clamped_y)) =
                    clamp_placement_to_boundary(x, y, geom_aabb, boundary_aabb)
                {
                    if clamped_y < best_y {
                        best_y = clamped_y;
                        best_placement = Some(PlacedItem {
                            instance_idx,
                            x: clamped_x,
                            y: clamped_y,
                            rotation,
                            score: clamped_y, // Score based on Y position
                        });
                    }
                }
            }
        }

        best_placement
    }

    /// Place items using BLF heuristic.
    fn place_items_blf(&self, items: &[usize], solution: &mut GdrrNestingSolution) {
        let margin = self.config.margin;
        let boundary_polygon = self.get_boundary_polygon_with_margin(margin);
        let sample_step = self.compute_sample_step();

        // Build placed geometries from current solution
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
                // Update solution
                let info = &self.instances[instance_idx];
                let area = self.geometry_areas[info.geometry_idx];

                solution.placed_area += area;
                solution.max_y = solution.max_y.max(placement.y);

                // Add to placed geometries for next iterations
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
}

impl GdrrProblem for GdrrNestingProblem {
    type Solution = GdrrNestingSolution;

    fn create_initial_solution(&mut self) -> GdrrNestingSolution {
        let boundary_area = self.boundary.measure();
        let mut solution = GdrrNestingSolution::new(self.instances.len(), boundary_area);

        // Place all items using BLF
        let all_items: Vec<usize> = (0..self.instances.len()).collect();
        self.place_items_blf(&all_items, &mut solution);

        solution
    }

    fn clone_solution(&self, solution: &GdrrNestingSolution) -> GdrrNestingSolution {
        solution.clone()
    }

    fn ruin_random(
        &mut self,
        solution: &mut GdrrNestingSolution,
        ratio: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> RuinResult {
        let num_to_remove = ((solution.placed.len() as f64 * ratio).ceil() as usize).max(1);
        let mut removed_items = Vec::new();

        if solution.placed.is_empty() {
            return RuinResult {
                removed_items,
                ruin_type: RuinType::Random,
            };
        }

        // Randomly select items to remove
        let mut indices: Vec<usize> = (0..solution.placed.len()).collect();
        indices.shuffle(rng);

        for &idx in indices.iter().take(num_to_remove) {
            let item = &solution.placed[idx];
            let info = &self.instances[item.instance_idx];

            removed_items.push(RuinedItem {
                index: item.instance_idx,
                geometry_id: self.geometries[info.geometry_idx].id().to_string(),
                position: vec![item.x, item.y],
                rotation: item.rotation,
                score: item.score,
            });
        }

        // Remove items from solution
        let removed_instance_indices: Vec<usize> = removed_items.iter().map(|r| r.index).collect();

        for idx in &removed_instance_indices {
            if let Some(pos) = solution.placed.iter().position(|p| p.instance_idx == *idx) {
                let item = solution.placed.remove(pos);
                let info = &self.instances[item.instance_idx];
                solution.placed_area -= self.geometry_areas[info.geometry_idx];
                solution.unplaced.push(item.instance_idx);
            }
        }

        // Recalculate max_y
        solution.max_y = solution.placed.iter().map(|p| p.y).fold(0.0, f64::max);

        RuinResult {
            removed_items,
            ruin_type: RuinType::Random,
        }
    }

    fn ruin_cluster(
        &mut self,
        solution: &mut GdrrNestingSolution,
        ratio: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> RuinResult {
        let num_to_remove = ((solution.placed.len() as f64 * ratio).ceil() as usize).max(1);
        let mut removed_items = Vec::new();

        if solution.placed.is_empty() {
            return RuinResult {
                removed_items,
                ruin_type: RuinType::Cluster,
            };
        }

        // Select a random seed item
        let seed_idx = rng.gen_range(0..solution.placed.len());
        let seed = &solution.placed[seed_idx];
        let seed_x = seed.x;
        let seed_y = seed.y;

        // Sort items by distance to seed
        let mut items_with_distance: Vec<(usize, f64)> = solution
            .placed
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let dx = item.x - seed_x;
                let dy = item.y - seed_y;
                (idx, (dx * dx + dy * dy).sqrt())
            })
            .collect();

        items_with_distance
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove closest items (including seed)
        for (idx, _) in items_with_distance.iter().take(num_to_remove) {
            let item = &solution.placed[*idx];
            let info = &self.instances[item.instance_idx];

            removed_items.push(RuinedItem {
                index: item.instance_idx,
                geometry_id: self.geometries[info.geometry_idx].id().to_string(),
                position: vec![item.x, item.y],
                rotation: item.rotation,
                score: item.score,
            });
        }

        // Remove items from solution
        let removed_instance_indices: Vec<usize> = removed_items.iter().map(|r| r.index).collect();

        for idx in &removed_instance_indices {
            if let Some(pos) = solution.placed.iter().position(|p| p.instance_idx == *idx) {
                let item = solution.placed.remove(pos);
                let info = &self.instances[item.instance_idx];
                solution.placed_area -= self.geometry_areas[info.geometry_idx];
                solution.unplaced.push(item.instance_idx);
            }
        }

        // Recalculate max_y
        solution.max_y = solution.placed.iter().map(|p| p.y).fold(0.0, f64::max);

        RuinResult {
            removed_items,
            ruin_type: RuinType::Cluster,
        }
    }

    fn ruin_worst(
        &mut self,
        solution: &mut GdrrNestingSolution,
        ratio: f64,
        _rng: &mut rand::rngs::StdRng,
    ) -> RuinResult {
        let num_to_remove = ((solution.placed.len() as f64 * ratio).ceil() as usize).max(1);
        let mut removed_items = Vec::new();

        if solution.placed.is_empty() {
            return RuinResult {
                removed_items,
                ruin_type: RuinType::Worst,
            };
        }

        // Sort by score (higher = worse, we use Y position as score)
        let mut items_with_score: Vec<(usize, f64)> = solution
            .placed
            .iter()
            .enumerate()
            .map(|(idx, item)| (idx, item.score))
            .collect();

        items_with_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove worst items
        for (idx, _) in items_with_score.iter().take(num_to_remove) {
            let item = &solution.placed[*idx];
            let info = &self.instances[item.instance_idx];

            removed_items.push(RuinedItem {
                index: item.instance_idx,
                geometry_id: self.geometries[info.geometry_idx].id().to_string(),
                position: vec![item.x, item.y],
                rotation: item.rotation,
                score: item.score,
            });
        }

        // Remove items from solution
        let removed_instance_indices: Vec<usize> = removed_items.iter().map(|r| r.index).collect();

        for idx in &removed_instance_indices {
            if let Some(pos) = solution.placed.iter().position(|p| p.instance_idx == *idx) {
                let item = solution.placed.remove(pos);
                let info = &self.instances[item.instance_idx];
                solution.placed_area -= self.geometry_areas[info.geometry_idx];
                solution.unplaced.push(item.instance_idx);
            }
        }

        // Recalculate max_y
        solution.max_y = solution.placed.iter().map(|p| p.y).fold(0.0, f64::max);

        RuinResult {
            removed_items,
            ruin_type: RuinType::Worst,
        }
    }

    fn recreate_best_fit(
        &mut self,
        solution: &mut GdrrNestingSolution,
        _ruined: &RuinResult,
    ) -> RecreateResult {
        let items_to_place = solution.unplaced.clone();
        let initial_placed = solution.placed.len();

        self.place_items_blf(&items_to_place, solution);

        RecreateResult {
            placed_count: solution.placed.len() - initial_placed,
            unplaced_count: solution.unplaced.len(),
            recreate_type: RecreateType::BestFit,
        }
    }

    fn recreate_blf(
        &mut self,
        solution: &mut GdrrNestingSolution,
        _ruined: &RuinResult,
    ) -> RecreateResult {
        let items_to_place = solution.unplaced.clone();
        let initial_placed = solution.placed.len();

        self.place_items_blf(&items_to_place, solution);

        RecreateResult {
            placed_count: solution.placed.len() - initial_placed,
            unplaced_count: solution.unplaced.len(),
            recreate_type: RecreateType::BottomLeftFill,
        }
    }

    fn recreate_nfp(
        &mut self,
        solution: &mut GdrrNestingSolution,
        _ruined: &RuinResult,
    ) -> RecreateResult {
        // NFP-guided placement (same as BLF but with potential for optimization)
        let items_to_place = solution.unplaced.clone();
        let initial_placed = solution.placed.len();

        self.place_items_blf(&items_to_place, solution);

        RecreateResult {
            placed_count: solution.placed.len() - initial_placed,
            unplaced_count: solution.unplaced.len(),
            recreate_type: RecreateType::NfpGuided,
        }
    }

    fn placement_score(&self, solution: &GdrrNestingSolution, item_index: usize) -> f64 {
        solution
            .placed
            .iter()
            .find(|p| p.instance_idx == item_index)
            .map(|p| p.score)
            .unwrap_or(f64::MAX)
    }

    fn get_neighbors(
        &self,
        solution: &GdrrNestingSolution,
        item_index: usize,
        radius: f64,
    ) -> Vec<usize> {
        let item = match solution
            .placed
            .iter()
            .find(|p| p.instance_idx == item_index)
        {
            Some(i) => i,
            None => return vec![],
        };

        solution
            .placed
            .iter()
            .filter(|p| {
                if p.instance_idx == item_index {
                    return false;
                }
                let dx = p.x - item.x;
                let dy = p.y - item.y;
                (dx * dx + dy * dy).sqrt() <= radius
            })
            .map(|p| p.instance_idx)
            .collect()
    }
}

/// Run GDRR nesting optimization.
pub fn run_gdrr_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    gdrr_config: &GdrrConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let mut problem = GdrrNestingProblem::new(
        geometries.to_vec(),
        boundary.clone(),
        config.clone(),
        cancelled,
        gdrr_config.time_limit_ms,
    );

    let runner = GdrrRunner::new(gdrr_config.clone());
    let gdrr_result: GdrrResult<GdrrNestingSolution> = runner.run(&mut problem, |_progress| {
        // Progress callback - can be used for logging
    });

    // Convert GDRR solution to SolveResult
    let mut result = SolveResult::new();

    for item in &gdrr_result.best_solution.placed {
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
    result.utilization = gdrr_result.best_solution.utilization();
    result.computation_time_ms = gdrr_result.elapsed_ms;
    result.iterations = Some(gdrr_result.iterations as u64);
    result.best_fitness = Some(gdrr_result.best_fitness);
    result.strategy = Some("GDRR".to_string());

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
    fn test_gdrr_nesting_problem_creation() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);

        assert_eq!(problem.num_instances(), 7); // 3 + 2 + 2
    }

    #[test]
    fn test_gdrr_nesting_initial_solution() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let solution = problem.create_initial_solution();

        assert!(solution.placed.len() > 0);
        assert!(solution.placed_area > 0.0);
    }

    #[test]
    fn test_gdrr_nesting_solution_fitness() {
        let solution = GdrrNestingSolution {
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
        // 1 unplaced item = 1000 penalty + utilization penalty + height penalty
        assert!(fitness >= 1000.0);
    }

    #[test]
    fn test_gdrr_nesting_ruin_random() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let initial_placed = solution.placed.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = problem.ruin_random(&mut solution, 0.3, &mut rng);

        assert!(!result.removed_items.is_empty());
        assert_eq!(result.ruin_type, RuinType::Random);
        assert!(solution.placed.len() < initial_placed);
    }

    #[test]
    fn test_gdrr_nesting_ruin_cluster() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let initial_placed = solution.placed.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = problem.ruin_cluster(&mut solution, 0.3, &mut rng);

        assert!(!result.removed_items.is_empty());
        assert_eq!(result.ruin_type, RuinType::Cluster);
        assert!(solution.placed.len() < initial_placed);
    }

    #[test]
    fn test_gdrr_nesting_ruin_worst() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let initial_placed = solution.placed.len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let result = problem.ruin_worst(&mut solution, 0.3, &mut rng);

        assert!(!result.removed_items.is_empty());
        assert_eq!(result.ruin_type, RuinType::Worst);
        assert!(solution.placed.len() < initial_placed);
    }

    #[test]
    fn test_gdrr_nesting_recreate() {
        use rand::SeedableRng;

        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);
        let mut solution = problem.create_initial_solution();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Ruin some items
        let ruin_result = problem.ruin_random(&mut solution, 0.5, &mut rng);
        let after_ruin_placed = solution.placed.len();

        // Recreate
        let recreate_result = problem.recreate_best_fit(&mut solution, &ruin_result);

        assert!(recreate_result.placed_count > 0 || after_ruin_placed == solution.placed.len());
        assert_eq!(recreate_result.recreate_type, RecreateType::BestFit);
    }

    #[test]
    fn test_run_gdrr_nesting() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let gdrr_config = GdrrConfig::new()
            .with_max_iterations(50)
            .with_time_limit_ms(5000);
        let cancelled = Arc::new(AtomicBool::new(false));

        let result = run_gdrr_nesting(&geometries, &boundary, &config, &gdrr_config, cancelled);

        assert!(!result.placements.is_empty());
        assert!(result.utilization > 0.0);
    }

    #[test]
    fn test_gdrr_nesting_full_cycle() {
        let geometries = create_test_geometries();
        let boundary = create_test_boundary();
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut problem = GdrrNestingProblem::new(geometries, boundary, config, cancelled, 60000);

        let gdrr_config = GdrrConfig::new().with_max_iterations(10).with_seed(42);

        let runner = GdrrRunner::new(gdrr_config);
        let result: GdrrResult<GdrrNestingSolution> = runner.run(&mut problem, |progress| {
            assert!(progress.iteration <= 10);
        });

        assert!(result.iterations <= 10);
        assert!(!result.best_solution.placed.is_empty());
    }
}
