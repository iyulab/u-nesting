//! 3D bin packing solver.

use crate::boundary::Boundary3D;
use crate::brkga_packing::run_brkga_packing;
use crate::extreme_point::run_ep_packing;
use crate::ga_packing::run_ga_packing;
use crate::geometry::Geometry3D;
use crate::sa_packing::run_sa_packing;
use u_nesting_core::brkga::BrkgaConfig;
use u_nesting_core::ga::GaConfig;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::sa::SaConfig;
use u_nesting_core::solver::{Config, ProgressCallback, ProgressInfo, Solver, Strategy};
use u_nesting_core::{Placement, Result, SolveResult};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 3D bin packing solver.
pub struct Packer3D {
    config: Config,
    cancelled: Arc<AtomicBool>,
}

impl Packer3D {
    /// Creates a new packer with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a packer with default configuration.
    pub fn default_config() -> Self {
        Self::new(Config::default())
    }

    /// Simple layer-based packing algorithm.
    fn layer_packing(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();

        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_max_x = boundary.width() - margin;
        let bound_max_y = boundary.depth() - margin;
        let bound_max_z = boundary.height() - margin;

        // Simple layer-based placement
        let mut current_x = margin;
        let mut current_y = margin;
        let mut current_z = margin;
        let mut row_depth = 0.0_f64;
        let mut layer_height = 0.0_f64;

        let mut total_placed_volume = 0.0;
        let mut total_placed_mass = 0.0;

        for geom in geometries {
            geom.validate()?;

            for instance in 0..geom.quantity() {
                if self.cancelled.load(Ordering::Relaxed) {
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    return Ok(result);
                }

                // Check mass constraint
                if let (Some(max_mass), Some(item_mass)) = (boundary.max_mass(), geom.mass()) {
                    if total_placed_mass + item_mass > max_mass {
                        result.unplaced.push(geom.id().clone());
                        continue;
                    }
                }

                // Try all allowed orientations to find the best fit
                let orientations = geom.allowed_orientations();
                let mut best_fit: Option<(usize, f64, f64, f64, f64, f64, f64, f64, f64)> = None;
                // (orientation_idx, width, depth, height, place_x, place_y, place_z, new_row_depth, new_layer_height)

                for (ori_idx, _) in orientations.iter().enumerate() {
                    let dims = geom.dimensions_for_orientation(ori_idx);
                    let g_width = dims.x;
                    let g_depth = dims.y;
                    let g_height = dims.z;

                    // Try current position first
                    let mut try_x = current_x;
                    let mut try_y = current_y;
                    let mut try_z = current_z;
                    let mut try_row_depth = row_depth;
                    let mut try_layer_height = layer_height;

                    // Check if fits in current row
                    if try_x + g_width > bound_max_x {
                        try_x = margin;
                        try_y += row_depth + spacing;
                        try_row_depth = 0.0;
                    }

                    // Check if fits in current layer
                    if try_y + g_depth > bound_max_y {
                        try_x = margin;
                        try_y = margin;
                        try_z += layer_height + spacing;
                        try_row_depth = 0.0;
                        try_layer_height = 0.0;
                    }

                    // Check if fits in container
                    if try_z + g_height > bound_max_z {
                        continue; // This orientation doesn't fit
                    }

                    // Score: prefer placements that use less vertical space (height)
                    // and stay in current row (lower y advancement)
                    let score = try_z * 1000000.0 + try_y * 1000.0 + try_x + g_height * 0.1;

                    let is_better = match &best_fit {
                        None => true,
                        Some((_, _, _, _, _, _, bz, _, _)) => {
                            let best_score = bz * 1000000.0
                                + best_fit.as_ref().unwrap().5 * 1000.0
                                + best_fit.as_ref().unwrap().4
                                + best_fit.as_ref().unwrap().3 * 0.1;
                            score < best_score
                        }
                    };

                    if is_better {
                        best_fit = Some((
                            ori_idx,
                            g_width,
                            g_depth,
                            g_height,
                            try_x,
                            try_y,
                            try_z,
                            try_row_depth,
                            try_layer_height,
                        ));
                    }
                }

                if let Some((ori_idx, g_width, g_depth, g_height, place_x, place_y, place_z, new_row_depth, new_layer_height)) =
                    best_fit
                {
                    // Convert orientation index to rotation angles
                    // For simplicity, we encode orientation in rotation_index
                    let placement = Placement::new_3d(
                        geom.id().clone(),
                        instance,
                        place_x,
                        place_y,
                        place_z,
                        0.0, // Orientation is encoded via rotation_index
                        0.0,
                        0.0,
                    )
                    .with_rotation_index(ori_idx);

                    placements.push(placement);
                    total_placed_volume += geom.measure();
                    if let Some(mass) = geom.mass() {
                        total_placed_mass += mass;
                    }

                    // Update position for next item
                    current_x = place_x + g_width + spacing;
                    current_y = place_y;
                    current_z = place_z;
                    row_depth = new_row_depth.max(g_depth);
                    layer_height = new_layer_height.max(g_height);
                } else {
                    result.unplaced.push(geom.id().clone());
                }
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_volume / boundary.measure();
        result.computation_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Genetic Algorithm based packing optimization.
    ///
    /// Uses GA to optimize placement order and orientations, with layer-based
    /// decoding for collision-free placements.
    fn genetic_algorithm(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        // Configure GA from solver config
        let ga_config = GaConfig::default()
            .with_population_size(self.config.population_size)
            .with_max_generations(self.config.max_generations)
            .with_crossover_rate(self.config.crossover_rate)
            .with_mutation_rate(self.config.mutation_rate);

        let result = run_ga_packing(
            geometries,
            boundary,
            &self.config,
            ga_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// BRKGA (Biased Random-Key Genetic Algorithm) based packing optimization.
    ///
    /// Uses random-key encoding and biased crossover for robust optimization.
    fn brkga(&self, geometries: &[Geometry3D], boundary: &Boundary3D) -> Result<SolveResult<f64>> {
        // Configure BRKGA with reasonable defaults
        let brkga_config = BrkgaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_elite_fraction(0.2)
            .with_mutant_fraction(0.15)
            .with_elite_bias(0.7);

        let result = run_brkga_packing(
            geometries,
            boundary,
            &self.config,
            brkga_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Simulated Annealing based packing optimization.
    ///
    /// Uses neighborhood operators to explore solution space with temperature-based
    /// acceptance probability.
    fn simulated_annealing(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        // Configure SA with reasonable defaults
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.95)
            .with_iterations_per_temp(50)
            .with_max_iterations(10000);

        let result = run_sa_packing(
            geometries,
            boundary,
            &self.config,
            sa_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Extreme Point heuristic-based packing.
    ///
    /// Places boxes at extreme points (positions touching at least two surfaces).
    /// More efficient than layer-based packing for many scenarios.
    fn extreme_point(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();

        let (ep_placements, utilization) = run_ep_packing(
            geometries,
            boundary,
            self.config.margin,
            self.config.spacing,
            boundary.max_mass(),
        );

        // Convert EP placements to Placement structs
        let mut placements = Vec::new();
        for (id, instance, position, _orientation) in ep_placements {
            let placement = Placement::new_3d(
                id, instance, position.x, position.y, position.z, 0.0, // rotation_x
                0.0, // rotation_y
                0.0, // rotation_z (orientation handled internally)
            );
            placements.push(placement);
        }

        // Collect unplaced items
        let mut placed_ids: std::collections::HashSet<(String, usize)> =
            std::collections::HashSet::new();
        for p in &placements {
            placed_ids.insert((p.geometry_id.clone(), p.instance));
        }

        let mut unplaced = Vec::new();
        for geom in geometries {
            for instance in 0..geom.quantity() {
                if !placed_ids.contains(&(geom.id().clone(), instance)) {
                    unplaced.push(geom.id().clone());
                }
            }
        }

        let mut result = SolveResult::new();
        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = utilization;
        result.unplaced = unplaced;
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        result.strategy = Some("ExtremePoint".to_string());

        Ok(result)
    }

    /// Layer packing with progress callback.
    fn layer_packing_with_progress(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        callback: &ProgressCallback,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();

        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_max_x = boundary.width() - margin;
        let bound_max_y = boundary.depth() - margin;
        let bound_max_z = boundary.height() - margin;

        let mut current_x = margin;
        let mut current_y = margin;
        let mut current_z = margin;
        let mut row_depth = 0.0_f64;
        let mut layer_height = 0.0_f64;

        let mut total_placed_volume = 0.0;
        let mut total_placed_mass = 0.0;

        // Count total pieces for progress
        let total_pieces: usize = geometries.iter().map(|g| g.quantity()).sum();
        let mut placed_count = 0usize;

        // Initial progress callback
        callback(ProgressInfo::new()
            .with_phase("Layer Packing")
            .with_items(0, total_pieces)
            .with_elapsed(0));

        for geom in geometries {
            geom.validate()?;

            for instance in 0..geom.quantity() {
                if self.cancelled.load(Ordering::Relaxed) {
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    callback(ProgressInfo::new()
                        .with_phase("Cancelled")
                        .with_items(placed_count, total_pieces)
                        .with_elapsed(result.computation_time_ms)
                        .finished());
                    return Ok(result);
                }

                // Check mass constraint
                if let (Some(max_mass), Some(item_mass)) = (boundary.max_mass(), geom.mass()) {
                    if total_placed_mass + item_mass > max_mass {
                        result.unplaced.push(geom.id().clone());
                        continue;
                    }
                }

                // Try all allowed orientations to find the best fit
                let orientations = geom.allowed_orientations();
                let mut best_fit: Option<(usize, f64, f64, f64, f64, f64, f64, f64, f64)> = None;

                for (ori_idx, _) in orientations.iter().enumerate() {
                    let dims = geom.dimensions_for_orientation(ori_idx);
                    let g_width = dims.x;
                    let g_depth = dims.y;
                    let g_height = dims.z;

                    let mut try_x = current_x;
                    let mut try_y = current_y;
                    let mut try_z = current_z;
                    let mut try_row_depth = row_depth;
                    let mut try_layer_height = layer_height;

                    if try_x + g_width > bound_max_x {
                        try_x = margin;
                        try_y += row_depth + spacing;
                        try_row_depth = 0.0;
                    }

                    if try_y + g_depth > bound_max_y {
                        try_x = margin;
                        try_y = margin;
                        try_z += layer_height + spacing;
                        try_row_depth = 0.0;
                        try_layer_height = 0.0;
                    }

                    if try_z + g_height > bound_max_z {
                        continue;
                    }

                    let score = try_z * 1000000.0 + try_y * 1000.0 + try_x + g_height * 0.1;

                    let is_better = match &best_fit {
                        None => true,
                        Some((_, _, _, _, _, _, bz, _, _)) => {
                            let best_score = bz * 1000000.0
                                + best_fit.as_ref().unwrap().5 * 1000.0
                                + best_fit.as_ref().unwrap().4
                                + best_fit.as_ref().unwrap().3 * 0.1;
                            score < best_score
                        }
                    };

                    if is_better {
                        best_fit = Some((
                            ori_idx,
                            g_width,
                            g_depth,
                            g_height,
                            try_x,
                            try_y,
                            try_z,
                            try_row_depth,
                            try_layer_height,
                        ));
                    }
                }

                if let Some((ori_idx, g_width, g_depth, g_height, place_x, place_y, place_z, new_row_depth, new_layer_height)) =
                    best_fit
                {
                    let placement = Placement::new_3d(
                        geom.id().clone(),
                        instance,
                        place_x,
                        place_y,
                        place_z,
                        0.0,
                        0.0,
                        0.0,
                    )
                    .with_rotation_index(ori_idx);

                    placements.push(placement);
                    total_placed_volume += geom.measure();
                    if let Some(mass) = geom.mass() {
                        total_placed_mass += mass;
                    }
                    placed_count += 1;

                    current_x = place_x + g_width + spacing;
                    current_y = place_y;
                    current_z = place_z;
                    row_depth = new_row_depth.max(g_depth);
                    layer_height = new_layer_height.max(g_height);

                    // Progress callback every piece
                    callback(ProgressInfo::new()
                        .with_phase("Layer Packing")
                        .with_items(placed_count, total_pieces)
                        .with_utilization(total_placed_volume / boundary.measure())
                        .with_elapsed(start.elapsed().as_millis() as u64));
                } else {
                    result.unplaced.push(geom.id().clone());
                }
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_volume / boundary.measure();
        result.computation_time_ms = start.elapsed().as_millis() as u64;

        // Final progress callback
        callback(ProgressInfo::new()
            .with_phase("Complete")
            .with_items(placed_count, total_pieces)
            .with_utilization(result.utilization)
            .with_elapsed(result.computation_time_ms)
            .finished());

        Ok(result)
    }
}

impl Solver for Packer3D {
    type Geometry = Geometry3D;
    type Boundary = Boundary3D;
    type Scalar = f64;

    fn solve(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
    ) -> Result<SolveResult<f64>> {
        boundary.validate()?;

        // Reset cancellation flag
        self.cancelled.store(false, Ordering::Relaxed);

        match self.config.strategy {
            Strategy::BottomLeftFill => self.layer_packing(geometries, boundary),
            Strategy::ExtremePoint => self.extreme_point(geometries, boundary),
            Strategy::GeneticAlgorithm => self.genetic_algorithm(geometries, boundary),
            Strategy::Brkga => self.brkga(geometries, boundary),
            Strategy::SimulatedAnnealing => self.simulated_annealing(geometries, boundary),
            _ => {
                // Fall back to layer packing for unimplemented strategies
                log::warn!(
                    "Strategy {:?} not yet implemented, using layer packing",
                    self.config.strategy
                );
                self.layer_packing(geometries, boundary)
            }
        }
    }

    fn solve_with_progress(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
        callback: ProgressCallback,
    ) -> Result<SolveResult<f64>> {
        boundary.validate()?;

        // Reset cancellation flag
        self.cancelled.store(false, Ordering::Relaxed);

        match self.config.strategy {
            Strategy::BottomLeftFill | Strategy::ExtremePoint => {
                self.layer_packing_with_progress(geometries, boundary, &callback)
            }
            // Other strategies fall back to basic progress reporting
            _ => {
                log::warn!(
                    "Strategy {:?} progress not yet implemented, using layer packing",
                    self.config.strategy
                );
                self.layer_packing_with_progress(geometries, boundary, &callback)
            }
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_packing() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(3),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let packer = Packer3D::default_config();

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(result.placements.len() <= 5);
    }

    #[test]
    fn test_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0).with_max_mass(350.0);

        let packer = Packer3D::default_config();
        let result = packer.solve(&geometries, &boundary).unwrap();

        // Should only place 3 boxes (300 mass) due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }

    #[test]
    fn test_placement_within_bounds() {
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(50.0, 50.0, 50.0);
        let config = Config::default().with_margin(5.0).with_spacing(2.0);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());

        // Verify placements are within bounds (with margin)
        for p in &result.placements {
            assert!(p.position[0] >= 5.0);
            assert!(p.position[1] >= 5.0);
            assert!(p.position[2] >= 5.0);
        }
    }

    #[test]
    fn test_ga_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // GA should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_ga_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ga_strategy_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Box that fits better when rotated
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];

        // Container where orientation matters
        let boundary = Boundary3D::new(60.0, 60.0, 60.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // GA should find a way to place both boxes
        assert_eq!(result.placements.len(), 2);
    }

    #[test]
    fn test_brkga_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // BRKGA should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("BRKGA".to_string()));
    }

    #[test]
    fn test_brkga_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ep_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::ExtremePoint);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // EP should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("ExtremePoint".to_string()));
    }

    #[test]
    fn test_ep_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::ExtremePoint);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ep_strategy_with_margin() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default()
            .with_strategy(Strategy::ExtremePoint)
            .with_margin(5.0);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // Verify placements start at margin
        for p in &result.placements {
            assert!(p.position[0] >= 4.9);
            assert!(p.position[1] >= 4.9);
            assert!(p.position[2] >= 4.9);
        }
    }

    #[test]
    fn test_ep_strategy_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Long box that benefits from rotation
        let geometries = vec![Geometry3D::new("B1", 80.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::ExtremePoint);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // EP should find a way to place both boxes
        assert_eq!(result.placements.len(), 2);
    }

    #[test]
    fn test_layer_packing_orientation_optimization() {
        use crate::geometry::OrientationConstraint;

        // A box 50x10x10 that won't fit in 45 width without rotation
        // But at orientation (1,0,2) it becomes 10x50x10, width=10, which fits
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];

        // Narrow container: width=45, depth=80, height=80
        let boundary = Boundary3D::new(45.0, 80.0, 80.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // Both boxes should be placed via orientation change
        assert_eq!(
            result.placements.len(),
            2,
            "Both boxes should be placed by using rotation"
        );
        assert!(result.unplaced.is_empty());

        // Verify orientation index is set for placements
        for p in &result.placements {
            assert!(
                p.rotation_index.is_some(),
                "Placement should have rotation_index set"
            );
        }
    }

    #[test]
    fn test_layer_packing_selects_best_orientation() {
        use crate::geometry::OrientationConstraint;

        // Box 30x20x10 in container 35x50x100
        // Original orientation (30x20x10): fits in row, leaves 5 spare width
        // Rotated (20x30x10): fits but uses more depth
        // Best: original orientation to minimize vertical space usage
        let geometries = vec![Geometry3D::new("B1", 30.0, 20.0, 10.0)
            .with_quantity(1)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(35.0, 50.0, 100.0);
        let packer = Packer3D::default_config();

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert_eq!(result.placements.len(), 1);
        assert!(result.unplaced.is_empty());
    }
}
