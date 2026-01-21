//! 2D nesting solver.

use crate::boundary::Boundary2D;
use crate::brkga_nesting::run_brkga_nesting;
use crate::ga_nesting::{run_ga_nesting, run_ga_nesting_with_progress};
use crate::geometry::Geometry2D;
use crate::nfp::{
    compute_ifp_with_margin, compute_nfp, find_bottom_left_placement, Nfp, NfpCache, PlacedGeometry,
};
use crate::sa_nesting::run_sa_nesting;
use u_nesting_core::brkga::BrkgaConfig;
use u_nesting_core::ga::GaConfig;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::sa::SaConfig;
use u_nesting_core::solver::{Config, ProgressCallback, ProgressInfo, Solver, Strategy};
use u_nesting_core::{Placement, Result, SolveResult};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 2D nesting solver.
pub struct Nester2D {
    config: Config,
    cancelled: Arc<AtomicBool>,
    #[allow(dead_code)] // Will be used for caching in future optimization
    nfp_cache: NfpCache,
}

impl Nester2D {
    /// Creates a new nester with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
            nfp_cache: NfpCache::new(),
        }
    }

    /// Creates a nester with default configuration.
    pub fn default_config() -> Self {
        Self::new(Config::default())
    }

    /// Bottom-Left Fill algorithm implementation with rotation optimization.
    fn bottom_left_fill(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();

        // Get boundary dimensions
        let (b_min, b_max) = boundary.aabb();
        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_min_x = b_min[0] + margin;
        let bound_min_y = b_min[1] + margin;
        let bound_max_x = b_max[0] - margin;
        let bound_max_y = b_max[1] - margin;

        let strip_width = bound_max_x - bound_min_x;
        let strip_height = bound_max_y - bound_min_y;

        // Simple row-based placement with rotation optimization
        let mut current_x = bound_min_x;
        let mut current_y = bound_min_y;
        let mut row_height = 0.0_f64;

        let mut total_placed_area = 0.0;

        for geom in geometries {
            geom.validate()?;

            // Get allowed rotation angles (default to 0 if none specified)
            let rotations = geom.rotations();
            let rotation_angles: Vec<f64> = if rotations.is_empty() {
                vec![0.0]
            } else {
                rotations
            };

            for instance in 0..geom.quantity() {
                if self.cancelled.load(Ordering::Relaxed) {
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    return Ok(result);
                }

                // Check time limit (0 = unlimited)
                if self.config.time_limit_ms > 0
                    && start.elapsed().as_millis() as u64 >= self.config.time_limit_ms
                {
                    result.boundaries_used = if placements.is_empty() { 0 } else { 1 };
                    result.utilization = total_placed_area / boundary.measure();
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    result.placements = placements;
                    return Ok(result);
                }

                // Find the best rotation for current position
                let mut best_fit: Option<(f64, f64, f64, f64, f64, [f64; 2])> = None; // (rotation, width, height, x, y, g_min)

                for &rotation in &rotation_angles {
                    let (g_min, g_max) = geom.aabb_at_rotation(rotation);
                    let g_width = g_max[0] - g_min[0];
                    let g_height = g_max[1] - g_min[1];

                    // Skip if geometry doesn't fit in boundary at all
                    if g_width > strip_width || g_height > strip_height {
                        continue;
                    }

                    // Calculate placement position for this rotation
                    let mut place_x = current_x;
                    let mut place_y = current_y;

                    // Check if piece fits in remaining row space
                    if place_x + g_width > bound_max_x {
                        // Would need to move to next row
                        place_x = bound_min_x;
                        place_y += row_height + spacing;
                    }

                    // Check if piece fits in boundary height
                    if place_y + g_height > bound_max_y {
                        continue; // This rotation doesn't fit
                    }

                    // Calculate score: prefer rotations that minimize wasted space
                    // Score = row advancement (lower is better)
                    let score = if place_x == bound_min_x && place_y > current_y {
                        // New row: score is based on new Y position
                        place_y - bound_min_y + g_height
                    } else {
                        // Same row: score is based on strip length advancement
                        place_x - bound_min_x + g_width
                    };

                    let is_better = match &best_fit {
                        None => true,
                        Some((_, _, _, _, _, _)) => {
                            // Prefer placements that don't start new rows
                            let best_score = if let Some((_, _, _, bx, by, _)) = best_fit {
                                if bx == bound_min_x && by > current_y {
                                    by - bound_min_y + g_height
                                } else {
                                    bx - bound_min_x + g_width
                                }
                            } else {
                                f64::INFINITY
                            };
                            score < best_score - 1e-6
                        }
                    };

                    if is_better {
                        best_fit = Some((rotation, g_width, g_height, place_x, place_y, g_min));
                    }
                }

                // Place the geometry with the best rotation
                if let Some((rotation, g_width, g_height, place_x, place_y, g_min)) = best_fit {
                    // Update row tracking if we moved to a new row
                    if place_x == bound_min_x && place_y > current_y {
                        row_height = 0.0;
                    }

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        instance,
                        place_x - g_min[0],
                        place_y - g_min[1],
                        rotation,
                    );

                    placements.push(placement);
                    total_placed_area += geom.measure();

                    // Update position for next piece
                    current_x = place_x + g_width + spacing;
                    current_y = place_y;
                    row_height = row_height.max(g_height);
                } else {
                    // Can't place this piece with any rotation
                    result.unplaced.push(geom.id().clone());
                }
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_area / boundary.measure();
        result.computation_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// NFP-guided Bottom-Left Fill algorithm.
    ///
    /// Uses No-Fit Polygons to find optimal placement positions that minimize
    /// wasted space while ensuring no overlaps.
    fn nfp_guided_blf(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();
        let mut placed_geometries: Vec<PlacedGeometry> = Vec::new();

        let margin = self.config.margin;
        let spacing = self.config.spacing;

        // Get boundary polygon with margin applied
        let boundary_polygon = self.get_boundary_polygon_with_margin(boundary, margin);

        let mut total_placed_area = 0.0;

        // Sampling step for grid search (adaptive based on geometry size)
        let sample_step = self.compute_sample_step(geometries);

        for geom in geometries {
            geom.validate()?;

            // Get allowed rotation angles
            let rotations = geom.rotations();
            let rotation_angles: Vec<f64> = if rotations.is_empty() {
                vec![0.0]
            } else {
                rotations
            };

            for instance in 0..geom.quantity() {
                if self.cancelled.load(Ordering::Relaxed) {
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    return Ok(result);
                }

                // Check time limit (0 = unlimited)
                if self.config.time_limit_ms > 0
                    && start.elapsed().as_millis() as u64 >= self.config.time_limit_ms
                {
                    result.boundaries_used = if placements.is_empty() { 0 } else { 1 };
                    result.utilization = total_placed_area / boundary.measure();
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    result.placements = placements;
                    return Ok(result);
                }

                // Try each rotation angle to find the best placement
                let mut best_placement: Option<(f64, f64, f64)> = None; // (x, y, rotation)

                for &rotation in &rotation_angles {
                    // Compute IFP for this geometry at this rotation (with margin from boundary)
                    let ifp =
                        match compute_ifp_with_margin(&boundary_polygon, geom, rotation, margin) {
                            Ok(ifp) => ifp,
                            Err(_) => continue,
                        };

                    if ifp.is_empty() {
                        continue;
                    }

                    // Compute NFPs with all placed geometries (using cache)
                    let mut nfps: Vec<Nfp> = Vec::new();
                    for placed in &placed_geometries {
                        // Use cache for NFP computation (between original geometries at origin)
                        // Key: (placed_geometry_id, current_geometry_id, rotation)
                        let cache_key = (
                            placed.geometry.id().as_str(),
                            geom.id().as_str(),
                            rotation - placed.rotation, // Relative rotation
                        );

                        // Compute NFP at origin and cache it
                        let nfp_at_origin =
                            match self.nfp_cache.get_or_compute(cache_key, || {
                                // Rotate placed geometry to its rotation
                                let placed_at_origin = placed.geometry.clone();
                                compute_nfp(&placed_at_origin, geom, rotation - placed.rotation)
                            }) {
                                Ok(nfp) => nfp,
                                Err(_) => continue,
                            };

                        // Translate cached NFP to placed position and expand by spacing
                        let translated_nfp = translate_nfp_exterior(&nfp_at_origin, placed.position);
                        let expanded = self.expand_nfp(&translated_nfp, spacing);
                        nfps.push(expanded);
                    }

                    // Shrink IFP by spacing from boundary
                    let ifp_shrunk = self.shrink_ifp(&ifp, spacing);

                    // Find the optimal valid placement (minimize X for shorter strip)
                    let nfp_refs: Vec<&Nfp> = nfps.iter().collect();
                    if let Some((x, y)) =
                        find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step)
                    {
                        // Compare with current best: prefer smaller X (shorter strip), then smaller Y
                        let is_better = match best_placement {
                            None => true,
                            Some((best_x, best_y, _)) => {
                                x < best_x - 1e-6 || (x < best_x + 1e-6 && y < best_y - 1e-6)
                            }
                        };
                        if is_better {
                            best_placement = Some((x, y, rotation));
                        }
                    }
                }

                // Place the geometry at the best position found
                if let Some((x, y, rotation)) = best_placement {
                    let placement = Placement::new_2d(geom.id().clone(), instance, x, y, rotation);

                    placements.push(placement);
                    placed_geometries.push(PlacedGeometry::new(geom.clone(), (x, y), rotation));
                    total_placed_area += geom.measure();
                } else {
                    // Could not place this instance
                    result.unplaced.push(geom.id().clone());
                }
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_area / boundary.measure();
        result.computation_time_ms = start.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Gets the boundary polygon with margin applied.
    fn get_boundary_polygon_with_margin(
        &self,
        boundary: &Boundary2D,
        margin: f64,
    ) -> Vec<(f64, f64)> {
        let (b_min, b_max) = boundary.aabb();

        // Create a rectangular boundary polygon with margin
        vec![
            (b_min[0] + margin, b_min[1] + margin),
            (b_max[0] - margin, b_min[1] + margin),
            (b_max[0] - margin, b_max[1] - margin),
            (b_min[0] + margin, b_max[1] - margin),
        ]
    }

    /// Computes an adaptive sample step based on geometry sizes.
    fn compute_sample_step(&self, geometries: &[Geometry2D]) -> f64 {
        if geometries.is_empty() {
            return 1.0;
        }

        // Use the smallest geometry dimension divided by 4 as sample step
        let mut min_dim = f64::INFINITY;
        for geom in geometries {
            let (g_min, g_max) = geom.aabb();
            let width = g_max[0] - g_min[0];
            let height = g_max[1] - g_min[1];
            min_dim = min_dim.min(width).min(height);
        }

        // Clamp sample step to reasonable range
        (min_dim / 4.0).clamp(0.5, 10.0)
    }

    /// Expands an NFP by the given spacing amount.
    fn expand_nfp(&self, nfp: &Nfp, spacing: f64) -> Nfp {
        if spacing <= 0.0 {
            return nfp.clone();
        }

        // Simple expansion: offset each polygon outward
        // For a proper implementation, this should use polygon offsetting
        // For now, we use a conservative approximation
        let expanded_polygons: Vec<Vec<(f64, f64)>> = nfp
            .polygons
            .iter()
            .map(|polygon| {
                // Compute centroid
                let (cx, cy) = polygon_centroid(polygon);

                // Offset each vertex away from centroid
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

        // Simple shrinking: offset each polygon inward
        let shrunk_polygons: Vec<Vec<(f64, f64)>> = ifp
            .polygons
            .iter()
            .filter_map(|polygon| {
                // Compute centroid
                let (cx, cy) = polygon_centroid(polygon);

                // Offset each vertex toward centroid
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
                            // Point too close to centroid, collapse to centroid
                            (cx, cy)
                        }
                    })
                    .collect();

                // Only keep polygon if it still has area
                if shrunk.len() >= 3 {
                    Some(shrunk)
                } else {
                    None
                }
            })
            .collect();

        Nfp::from_polygons(shrunk_polygons)
    }

    /// Genetic Algorithm based nesting optimization.
    ///
    /// Uses GA to optimize placement order and rotations, with NFP-guided
    /// decoding for collision-free placements.
    fn genetic_algorithm(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        // Configure GA from solver config
        let ga_config = GaConfig::default()
            .with_population_size(self.config.population_size)
            .with_max_generations(self.config.max_generations)
            .with_crossover_rate(self.config.crossover_rate)
            .with_mutation_rate(self.config.mutation_rate);

        let result = run_ga_nesting(
            geometries,
            boundary,
            &self.config,
            ga_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// BRKGA (Biased Random-Key Genetic Algorithm) based nesting optimization.
    ///
    /// Uses random-key encoding and biased crossover for robust optimization.
    fn brkga(&self, geometries: &[Geometry2D], boundary: &Boundary2D) -> Result<SolveResult<f64>> {
        // Configure BRKGA with reasonable defaults
        let brkga_config = BrkgaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_elite_fraction(0.2)
            .with_mutant_fraction(0.15)
            .with_elite_bias(0.7);

        let result = run_brkga_nesting(
            geometries,
            boundary,
            &self.config,
            brkga_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Simulated Annealing based nesting optimization.
    ///
    /// Uses neighborhood operators to explore solution space with temperature-based
    /// acceptance probability.
    fn simulated_annealing(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        // Configure SA with reasonable defaults
        let sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.95)
            .with_iterations_per_temp(50)
            .with_max_iterations(10000);

        let result = run_sa_nesting(
            geometries,
            boundary,
            &self.config,
            sa_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Bottom-Left Fill with progress callback.
    fn bottom_left_fill_with_progress(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
        callback: &ProgressCallback,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();

        // Get boundary dimensions
        let (b_min, b_max) = boundary.aabb();
        let margin = self.config.margin;
        let spacing = self.config.spacing;

        let bound_min_x = b_min[0] + margin;
        let bound_min_y = b_min[1] + margin;
        let bound_max_x = b_max[0] - margin;
        let bound_max_y = b_max[1] - margin;

        let strip_width = bound_max_x - bound_min_x;
        let strip_height = bound_max_y - bound_min_y;

        let mut current_x = bound_min_x;
        let mut current_y = bound_min_y;
        let mut row_height = 0.0_f64;
        let mut total_placed_area = 0.0;

        // Count total pieces for progress
        let total_pieces: usize = geometries.iter().map(|g| g.quantity()).sum();
        let mut placed_count = 0usize;

        // Initial progress callback
        callback(ProgressInfo::new()
            .with_phase("BLF Placement")
            .with_items(0, total_pieces)
            .with_elapsed(0));

        for geom in geometries {
            geom.validate()?;

            let rotations = geom.rotations();
            let rotation_angles: Vec<f64> = if rotations.is_empty() {
                vec![0.0]
            } else {
                rotations
            };

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

                // Check time limit (0 = unlimited)
                if self.config.time_limit_ms > 0
                    && start.elapsed().as_millis() as u64 >= self.config.time_limit_ms
                {
                    result.boundaries_used = if placements.is_empty() { 0 } else { 1 };
                    result.utilization = total_placed_area / boundary.measure();
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    result.placements = placements;
                    callback(ProgressInfo::new()
                        .with_phase("Time Limit Reached")
                        .with_items(placed_count, total_pieces)
                        .with_elapsed(result.computation_time_ms)
                        .finished());
                    return Ok(result);
                }

                let mut best_fit: Option<(f64, f64, f64, f64, f64, [f64; 2])> = None;

                for &rotation in &rotation_angles {
                    let (g_min, g_max) = geom.aabb_at_rotation(rotation);
                    let g_width = g_max[0] - g_min[0];
                    let g_height = g_max[1] - g_min[1];

                    if g_width > strip_width || g_height > strip_height {
                        continue;
                    }

                    let mut place_x = current_x;
                    let mut place_y = current_y;

                    if place_x + g_width > bound_max_x {
                        place_x = bound_min_x;
                        place_y += row_height + spacing;
                    }

                    if place_y + g_height > bound_max_y {
                        continue;
                    }

                    let score = if place_x == bound_min_x && place_y > current_y {
                        place_y - bound_min_y + g_height
                    } else {
                        place_x - bound_min_x + g_width
                    };

                    let is_better = match &best_fit {
                        None => true,
                        Some((_, _, _, bx, by, _)) => {
                            let best_score = if *bx == bound_min_x && *by > current_y {
                                by - bound_min_y
                            } else {
                                bx - bound_min_x
                            };
                            score < best_score - 1e-6
                        }
                    };

                    if is_better {
                        best_fit = Some((rotation, g_width, g_height, place_x, place_y, g_min));
                    }
                }

                if let Some((rotation, g_width, g_height, place_x, place_y, g_min)) = best_fit {
                    if place_x == bound_min_x && place_y > current_y {
                        row_height = 0.0;
                    }

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        instance,
                        place_x - g_min[0],
                        place_y - g_min[1],
                        rotation,
                    );

                    placements.push(placement);
                    total_placed_area += geom.measure();
                    placed_count += 1;

                    current_x = place_x + g_width + spacing;
                    current_y = place_y;
                    row_height = row_height.max(g_height);

                    // Progress callback every piece
                    callback(ProgressInfo::new()
                        .with_phase("BLF Placement")
                        .with_items(placed_count, total_pieces)
                        .with_utilization(total_placed_area / boundary.measure())
                        .with_elapsed(start.elapsed().as_millis() as u64));
                } else {
                    result.unplaced.push(geom.id().clone());
                }
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_area / boundary.measure();
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

    /// NFP-guided BLF with progress callback.
    fn nfp_guided_blf_with_progress(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
        callback: &ProgressCallback,
    ) -> Result<SolveResult<f64>> {
        let start = Instant::now();
        let mut result = SolveResult::new();
        let mut placements = Vec::new();
        let mut placed_geometries: Vec<PlacedGeometry> = Vec::new();

        let margin = self.config.margin;
        let spacing = self.config.spacing;
        let boundary_polygon = self.get_boundary_polygon_with_margin(boundary, margin);

        let mut total_placed_area = 0.0;
        let sample_step = self.compute_sample_step(geometries);

        // Count total pieces for progress
        let total_pieces: usize = geometries.iter().map(|g| g.quantity()).sum();
        let mut placed_count = 0usize;

        // Initial progress callback
        callback(ProgressInfo::new()
            .with_phase("NFP Placement")
            .with_items(0, total_pieces)
            .with_elapsed(0));

        for geom in geometries {
            geom.validate()?;

            let rotations = geom.rotations();
            let rotation_angles: Vec<f64> = if rotations.is_empty() {
                vec![0.0]
            } else {
                rotations
            };

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

                // Check time limit (0 = unlimited)
                if self.config.time_limit_ms > 0
                    && start.elapsed().as_millis() as u64 >= self.config.time_limit_ms
                {
                    result.boundaries_used = if placements.is_empty() { 0 } else { 1 };
                    result.utilization = total_placed_area / boundary.measure();
                    result.computation_time_ms = start.elapsed().as_millis() as u64;
                    result.placements = placements;
                    callback(ProgressInfo::new()
                        .with_phase("Time Limit Reached")
                        .with_items(placed_count, total_pieces)
                        .with_elapsed(result.computation_time_ms)
                        .finished());
                    return Ok(result);
                }

                let mut best_placement: Option<(f64, f64, f64)> = None;

                for &rotation in &rotation_angles {
                    let ifp = match compute_ifp_with_margin(&boundary_polygon, geom, rotation, margin) {
                        Ok(ifp) => ifp,
                        Err(_) => continue,
                    };

                    if ifp.is_empty() {
                        continue;
                    }

                    let mut nfps: Vec<Nfp> = Vec::new();
                    for placed in &placed_geometries {
                        // Use cache for NFP computation
                        let cache_key = (
                            placed.geometry.id().as_str(),
                            geom.id().as_str(),
                            rotation - placed.rotation,
                        );

                        let nfp_at_origin =
                            match self.nfp_cache.get_or_compute(cache_key, || {
                                let placed_at_origin = placed.geometry.clone();
                                compute_nfp(&placed_at_origin, geom, rotation - placed.rotation)
                            }) {
                                Ok(nfp) => nfp,
                                Err(_) => continue,
                            };

                        let translated_nfp = translate_nfp_exterior(&nfp_at_origin, placed.position);
                        let expanded = self.expand_nfp(&translated_nfp, spacing);
                        nfps.push(expanded);
                    }

                    let ifp_shrunk = self.shrink_ifp(&ifp, spacing);
                    let nfp_refs: Vec<&Nfp> = nfps.iter().collect();

                    if let Some((x, y)) = find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step) {
                        let is_better = match best_placement {
                            None => true,
                            Some((best_x, best_y, _)) => {
                                x < best_x - 1e-6 || (x < best_x + 1e-6 && y < best_y - 1e-6)
                            }
                        };
                        if is_better {
                            best_placement = Some((x, y, rotation));
                        }
                    }
                }

                if let Some((x, y, rotation)) = best_placement {
                    let placement = Placement::new_2d(geom.id().clone(), instance, x, y, rotation);
                    placements.push(placement);
                    placed_geometries.push(PlacedGeometry::new(geom.clone(), (x, y), rotation));
                    total_placed_area += geom.measure();
                    placed_count += 1;

                    // Progress callback every piece
                    callback(ProgressInfo::new()
                        .with_phase("NFP Placement")
                        .with_items(placed_count, total_pieces)
                        .with_utilization(total_placed_area / boundary.measure())
                        .with_elapsed(start.elapsed().as_millis() as u64));
                } else {
                    result.unplaced.push(geom.id().clone());
                }
            }
        }

        result.placements = placements;
        result.boundaries_used = 1;
        result.utilization = total_placed_area / boundary.measure();
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

/// Translates an NFP by the given offset.
fn translate_nfp_exterior(nfp: &Nfp, offset: (f64, f64)) -> Nfp {
    Nfp {
        polygons: nfp
            .polygons
            .iter()
            .map(|polygon| {
                polygon
                    .iter()
                    .map(|(x, y)| (x + offset.0, y + offset.1))
                    .collect()
            })
            .collect(),
    }
}

impl Solver for Nester2D {
    type Geometry = Geometry2D;
    type Boundary = Boundary2D;
    type Scalar = f64;

    fn solve(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
    ) -> Result<SolveResult<f64>> {
        boundary.validate()?;

        // Reset cancellation flag
        self.cancelled.store(false, Ordering::Relaxed);

        let mut result = match self.config.strategy {
            Strategy::BottomLeftFill => self.bottom_left_fill(geometries, boundary),
            Strategy::NfpGuided => self.nfp_guided_blf(geometries, boundary),
            Strategy::GeneticAlgorithm => self.genetic_algorithm(geometries, boundary),
            Strategy::Brkga => self.brkga(geometries, boundary),
            Strategy::SimulatedAnnealing => self.simulated_annealing(geometries, boundary),
            _ => {
                // Fall back to NFP-guided BLF for other strategies
                log::warn!(
                    "Strategy {:?} not yet implemented, using NfpGuided",
                    self.config.strategy
                );
                self.nfp_guided_blf(geometries, boundary)
            }
        }?;

        // Remove duplicate entries from unplaced list
        result.deduplicate_unplaced();
        Ok(result)
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

        let mut result = match self.config.strategy {
            Strategy::BottomLeftFill => {
                self.bottom_left_fill_with_progress(geometries, boundary, &callback)?
            }
            Strategy::NfpGuided => {
                self.nfp_guided_blf_with_progress(geometries, boundary, &callback)?
            }
            Strategy::GeneticAlgorithm => {
                let ga_config = GaConfig::default()
                    .with_population_size(self.config.population_size)
                    .with_max_generations(self.config.max_generations)
                    .with_crossover_rate(self.config.crossover_rate)
                    .with_mutation_rate(self.config.mutation_rate);

                run_ga_nesting_with_progress(
                    geometries,
                    boundary,
                    &self.config,
                    ga_config,
                    self.cancelled.clone(),
                    callback,
                )
            }
            // For other strategies, use basic progress reporting
            _ => {
                log::warn!(
                    "Strategy {:?} not yet implemented, using NfpGuided",
                    self.config.strategy
                );
                self.nfp_guided_blf_with_progress(geometries, boundary, &callback)?
            }
        };

        // Remove duplicate entries from unplaced list
        result.deduplicate_unplaced();
        Ok(result)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_nesting() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(3),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let nester = Nester2D::default_config();

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(result.placements.len() <= 5); // 3 + 2 = 5 pieces
    }

    #[test]
    fn test_placement_within_bounds() {
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default().with_margin(5.0).with_spacing(2.0);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All pieces should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());

        // Verify placements are within bounds (with margin)
        for p in &result.placements {
            assert!(p.position[0] >= 5.0);
            assert!(p.position[1] >= 5.0);
        }
    }

    #[test]
    fn test_nfp_guided_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(1),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default().with_strategy(Strategy::NfpGuided);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert_eq!(result.placements.len(), 3); // 2 + 1 = 3 pieces
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_nfp_guided_with_spacing() {
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default()
            .with_strategy(Strategy::NfpGuided)
            .with_margin(2.0)
            .with_spacing(3.0);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All pieces should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());

        // Utilization should be positive
        assert!(result.utilization > 0.0);
    }

    #[test]
    fn test_nfp_guided_no_overlap() {
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(3)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::NfpGuided);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert_eq!(result.placements.len(), 3);

        // Verify no overlaps between placements
        for i in 0..result.placements.len() {
            for j in (i + 1)..result.placements.len() {
                let p1 = &result.placements[i];
                let p2 = &result.placements[j];

                // Simple AABB overlap check for rectangles
                let r1_min_x = p1.position[0];
                let r1_max_x = p1.position[0] + 20.0;
                let r1_min_y = p1.position[1];
                let r1_max_y = p1.position[1] + 20.0;

                let r2_min_x = p2.position[0];
                let r2_max_x = p2.position[0] + 20.0;
                let r2_min_y = p2.position[1];
                let r2_max_y = p2.position[1] + 20.0;

                // Check no overlap (with small tolerance for floating point)
                let overlaps_x = r1_min_x < r2_max_x - 0.01 && r1_max_x > r2_min_x + 0.01;
                let overlaps_y = r1_min_y < r2_max_y - 0.01 && r1_max_y > r2_min_y + 0.01;

                assert!(
                    !(overlaps_x && overlaps_y),
                    "Placements {} and {} overlap",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_nfp_guided_utilization() {
        // Perfect fit: 4 rectangles of 25x25 in a 100x50 boundary
        let geometries = vec![Geometry2D::rectangle("R1", 25.0, 25.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default().with_strategy(Strategy::NfpGuided);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All pieces should be placed
        assert_eq!(result.placements.len(), 4);

        // Utilization should be 50% (4 * 625 = 2500 / 5000)
        assert!(result.utilization > 0.45);
    }

    #[test]
    fn test_polygon_centroid() {
        // Test the centroid calculation
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (cx, cy) = polygon_centroid(&square);
        assert!((cx - 5.0).abs() < 0.01);
        assert!((cy - 5.0).abs() < 0.01);

        let triangle = vec![(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)];
        let (cx, cy) = polygon_centroid(&triangle);
        assert!((cx - 3.0).abs() < 0.01);
        assert!((cy - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_ga_strategy_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        // GA should report generations and fitness
        assert!(result.generations.is_some());
        assert!(result.best_fitness.is_some());
        assert!(result.strategy == Some("GeneticAlgorithm".to_string()));
    }

    #[test]
    fn test_ga_strategy_all_placed() {
        // Easy case: 4 small rectangles in large boundary
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All 4 pieces should fit
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_brkga_strategy_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        // BRKGA should report generations and fitness
        assert!(result.generations.is_some());
        assert!(result.best_fitness.is_some());
        assert!(result.strategy == Some("BRKGA".to_string()));
    }

    #[test]
    fn test_brkga_strategy_all_placed() {
        // Easy case: 4 small rectangles in large boundary
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All 4 pieces should fit
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_blf_rotation_optimization() {
        // Test that BLF uses rotation to optimize placement
        // A 30x10 rectangle can fit better in a narrow strip when rotated 90 degrees
        let geometries = vec![
            Geometry2D::rectangle("R1", 30.0, 10.0)
                .with_rotations(vec![0.0, std::f64::consts::FRAC_PI_2]) // 0 and 90 degrees
                .with_quantity(3),
        ];

        // Strip that's 35 wide: 30x10 won't fit two side-by-side at 0 deg
        // But two 10x30 (rotated 90 deg) can fit vertically in 95 height
        let boundary = Boundary2D::rectangle(35.0, 95.0);
        let nester = Nester2D::default_config();

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All 3 pieces should be placed (by rotating)
        assert_eq!(
            result.placements.len(),
            3,
            "All pieces should be placed with rotation optimization"
        );
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_blf_selects_best_rotation() {
        // Verify BLF selects optimal rotation, not just the first one
        let geometries = vec![
            Geometry2D::rectangle("R1", 40.0, 10.0)
                .with_rotations(vec![0.0, std::f64::consts::FRAC_PI_2]) // 0 and 90 degrees
                .with_quantity(2),
        ];

        // In a 45x50 boundary:
        // - At 0 deg: 40x10, only one fits horizontally (40 < 45), next row needed
        // - At 90 deg: 10x40, two can fit side-by-side (10+10 < 45) in one row
        let boundary = Boundary2D::rectangle(45.0, 50.0);
        let nester = Nester2D::default_config();

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert_eq!(result.placements.len(), 2);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_progress_callback_blf() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(4)];
        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();
        let last_items_placed = Arc::new(AtomicUsize::new(0));
        let last_items_placed_clone = last_items_placed.clone();

        let callback: ProgressCallback = Box::new(move |info| {
            callback_count_clone.fetch_add(1, Ordering::Relaxed);
            last_items_placed_clone.store(info.items_placed, Ordering::Relaxed);
        });

        let result = nester.solve_with_progress(&geometries, &boundary, callback).unwrap();

        // Verify callback was called (at least once per piece + initial + final)
        let count = callback_count.load(Ordering::Relaxed);
        assert!(count >= 5, "Expected at least 5 callbacks (1 initial + 4 pieces + 1 final), got {}", count);

        // Verify final items_placed
        let final_placed = last_items_placed.load(Ordering::Relaxed);
        assert_eq!(final_placed, 4, "Should report 4 items placed");

        // Verify result
        assert_eq!(result.placements.len(), 4);
    }

    #[test]
    fn test_progress_callback_nfp() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(2)];
        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default().with_strategy(Strategy::NfpGuided);
        let nester = Nester2D::new(config);

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();

        let callback: ProgressCallback = Box::new(move |info| {
            callback_count_clone.fetch_add(1, Ordering::Relaxed);
            assert!(info.items_placed <= info.total_items);
        });

        let result = nester.solve_with_progress(&geometries, &boundary, callback).unwrap();

        // Verify callback was called
        let count = callback_count.load(Ordering::Relaxed);
        assert!(count >= 3, "Expected at least 3 callbacks, got {}", count);

        // Verify result
        assert_eq!(result.placements.len(), 2);
    }

    #[test]
    fn test_time_limit_honored() {
        // Create many geometries to ensure BLF takes measurable time
        let geometries: Vec<Geometry2D> = (0..100)
            .map(|i| Geometry2D::rectangle(&format!("R{}", i), 5.0, 5.0))
            .collect();
        let boundary = Boundary2D::rectangle(1000.0, 1000.0);

        // Set a very short time limit (1ms) to ensure timeout
        let config = Config::default()
            .with_strategy(Strategy::BottomLeftFill)
            .with_time_limit(1);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // With such a short time limit, we may not place all items
        // The test verifies that the solver respects the time limit
        assert!(
            result.computation_time_ms <= 100, // Allow some margin for overhead
            "Computation took too long: {}ms (expected <= 100ms with 1ms limit)",
            result.computation_time_ms
        );
    }

    #[test]
    fn test_time_limit_zero_unlimited() {
        // time_limit_ms = 0 means unlimited
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(4)];
        let boundary = Boundary2D::rectangle(50.0, 50.0);

        let config = Config::default()
            .with_strategy(Strategy::BottomLeftFill)
            .with_time_limit(0); // Unlimited
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // Should place all items (no early exit)
        assert_eq!(result.placements.len(), 4);
    }
}
