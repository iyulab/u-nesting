//! 2D nesting solver.

use crate::alns_nesting::run_alns_nesting;
use crate::boundary::Boundary2D;
use crate::brkga_nesting::run_brkga_nesting;
use crate::ga_nesting::{run_ga_nesting, run_ga_nesting_with_progress};
use crate::gdrr_nesting::run_gdrr_nesting;
use crate::geometry::Geometry2D;
#[cfg(feature = "milp")]
use crate::milp_solver::run_milp_nesting;
use crate::nfp::{
    compute_ifp_with_margin, compute_nfp, find_bottom_left_placement, rotate_nfp, translate_nfp,
    Nfp, NfpCache, PlacedGeometry,
};
#[cfg(feature = "milp")]
#[allow(unused_imports)]
use crate::nfp_cm_solver::run_nfp_cm_nesting;
use crate::sa_nesting::run_sa_nesting;
use u_nesting_core::alns::AlnsConfig;
use u_nesting_core::brkga::BrkgaConfig;
#[cfg(feature = "milp")]
use u_nesting_core::exact::ExactConfig;
use u_nesting_core::ga::GaConfig;
use u_nesting_core::gdrr::GdrrConfig;
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

                    // Compute origin position from AABB position
                    let origin_x = place_x - g_min[0];
                    let origin_y = place_y - g_min[1];

                    // Compute valid position bounds and clamp to ensure geometry stays within boundary
                    // The position represents the geometry's origin (where local (0,0) is placed).
                    // For geometries with g_min > 0, the leftmost point is at origin + g_min.
                    // To ensure the origin position is >= boundary min (important for multi-strip
                    // scenarios where viewers determine strip by origin position), we use
                    // max(calculated_min, boundary_min) for the minimum valid position.
                    let (g_min_rot, g_max_rot) = geom.aabb_at_rotation(rotation);
                    let min_valid_x = (b_min[0] + margin - g_min_rot[0]).max(b_min[0] + margin);
                    let max_valid_x = b_max[0] - margin - g_max_rot[0];
                    let min_valid_y = (b_min[1] + margin - g_min_rot[1]).max(b_min[1] + margin);
                    let max_valid_y = b_max[1] - margin - g_max_rot[1];

                    let clamped_x = origin_x.clamp(min_valid_x, max_valid_x);
                    let clamped_y = origin_y.clamp(min_valid_y, max_valid_y);

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        instance,
                        clamped_x,
                        clamped_y,
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

                        // Compute NFP at origin and cache it (with relative rotation)
                        // NFP is computed between the placed geometry at origin (no rotation)
                        // and the new geometry with relative rotation applied.
                        // Formula: NFP_actual = translate(rotate(NFP_relative, placed.rotation), placed.position)
                        let nfp_at_origin = match self.nfp_cache.get_or_compute(cache_key, || {
                            let placed_at_origin = placed.geometry.clone();
                            compute_nfp(&placed_at_origin, geom, rotation - placed.rotation)
                        }) {
                            Ok(nfp) => nfp,
                            Err(_) => continue,
                        };

                        // Transform NFP: first rotate by placed.rotation, then translate to placed.position
                        // This correctly accounts for the placed geometry's actual orientation
                        let rotated_nfp = rotate_nfp(&nfp_at_origin, placed.rotation);
                        let translated_nfp = translate_nfp(&rotated_nfp, placed.position);
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
                    // Compute valid position bounds and clamp to ensure geometry stays within boundary
                    // Ensure origin position >= boundary min for multi-strip compatibility
                    let (b_min, b_max) = boundary.aabb();
                    let (g_min, g_max) = geom.aabb_at_rotation(rotation);
                    let min_valid_x = (b_min[0] + margin - g_min[0]).max(b_min[0] + margin);
                    let max_valid_x = b_max[0] - margin - g_max[0];
                    let min_valid_y = (b_min[1] + margin - g_min[1]).max(b_min[1] + margin);
                    let max_valid_y = b_max[1] - margin - g_max[1];

                    let clamped_x = x.clamp(min_valid_x, max_valid_x);
                    let clamped_y = y.clamp(min_valid_y, max_valid_y);

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        instance,
                        clamped_x,
                        clamped_y,
                        rotation,
                    );

                    placements.push(placement);
                    placed_geometries.push(PlacedGeometry::new(
                        geom.clone(),
                        (clamped_x, clamped_y),
                        rotation,
                    ));
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
        let mut ga_config = GaConfig::default()
            .with_population_size(self.config.population_size)
            .with_max_generations(self.config.max_generations)
            .with_crossover_rate(self.config.crossover_rate)
            .with_mutation_rate(self.config.mutation_rate);

        // Apply time limit if specified
        if self.config.time_limit_ms > 0 {
            ga_config = ga_config
                .with_time_limit(std::time::Duration::from_millis(self.config.time_limit_ms));
        }

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
        let mut brkga_config = BrkgaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_elite_fraction(0.2)
            .with_mutant_fraction(0.15)
            .with_elite_bias(0.7);

        // Apply time limit if specified
        if self.config.time_limit_ms > 0 {
            brkga_config = brkga_config
                .with_time_limit(std::time::Duration::from_millis(self.config.time_limit_ms));
        }

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
        let mut sa_config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.95)
            .with_iterations_per_temp(50)
            .with_max_iterations(10000);

        // Apply time limit if specified
        if self.config.time_limit_ms > 0 {
            sa_config = sa_config
                .with_time_limit(std::time::Duration::from_millis(self.config.time_limit_ms));
        }

        let result = run_sa_nesting(
            geometries,
            boundary,
            &self.config,
            sa_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Goal-Driven Ruin and Recreate (GDRR) optimization.
    fn gdrr(&self, geometries: &[Geometry2D], boundary: &Boundary2D) -> Result<SolveResult<f64>> {
        // Configure GDRR with reasonable defaults
        // Use user's time limit, default to 60s if not specified
        let time_limit = if self.config.time_limit_ms > 0 {
            self.config.time_limit_ms
        } else {
            60000
        };
        let gdrr_config = GdrrConfig::default()
            .with_max_iterations(5000)
            .with_time_limit_ms(time_limit)
            .with_ruin_ratio(0.1, 0.4)
            .with_lahc_list_length(50);

        let result = run_gdrr_nesting(
            geometries,
            boundary,
            &self.config,
            &gdrr_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Adaptive Large Neighborhood Search (ALNS) optimization.
    fn alns(&self, geometries: &[Geometry2D], boundary: &Boundary2D) -> Result<SolveResult<f64>> {
        // Configure ALNS with reasonable defaults
        // Use user's time limit, default to 60s if not specified
        let time_limit = if self.config.time_limit_ms > 0 {
            self.config.time_limit_ms
        } else {
            60000
        };
        let alns_config = AlnsConfig::default()
            .with_max_iterations(5000)
            .with_time_limit_ms(time_limit)
            .with_segment_size(100)
            .with_scores(33.0, 9.0, 13.0)
            .with_reaction_factor(0.1)
            .with_temperature(100.0, 0.9995, 0.1);

        let result = run_alns_nesting(
            geometries,
            boundary,
            &self.config,
            &alns_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// MILP-based exact solver.
    #[cfg(feature = "milp")]
    fn milp_exact(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(self.config.time_limit_ms.max(60000))
            .with_max_items(15)
            .with_rotation_steps(4)
            .with_grid_step(1.0);

        let result = run_milp_nesting(
            geometries,
            boundary,
            &self.config,
            &exact_config,
            self.cancelled.clone(),
        );

        Ok(result)
    }

    /// Hybrid exact solver: try MILP first, fallback to heuristic.
    #[cfg(feature = "milp")]
    fn hybrid_exact(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        // Count total instances
        let total_instances: usize = geometries.iter().map(|g| g.quantity()).sum();

        // If small enough, try exact
        if total_instances <= 15 {
            let exact_config = ExactConfig::default()
                .with_time_limit_ms((self.config.time_limit_ms / 2).max(30000))
                .with_max_items(15);

            let exact_result = run_milp_nesting(
                geometries,
                boundary,
                &self.config,
                &exact_config,
                self.cancelled.clone(),
            );

            // If got a good solution, return it
            if !exact_result.placements.is_empty() {
                return Ok(exact_result);
            }
        }

        // Fallback to ALNS (best heuristic)
        self.alns(geometries, boundary)
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
        callback(
            ProgressInfo::new()
                .with_phase("BLF Placement")
                .with_items(0, total_pieces)
                .with_elapsed(0),
        );

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
                    callback(
                        ProgressInfo::new()
                            .with_phase("Cancelled")
                            .with_items(placed_count, total_pieces)
                            .with_elapsed(result.computation_time_ms)
                            .finished(),
                    );
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
                    callback(
                        ProgressInfo::new()
                            .with_phase("Time Limit Reached")
                            .with_items(placed_count, total_pieces)
                            .with_elapsed(result.computation_time_ms)
                            .finished(),
                    );
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

                    // Compute origin position from AABB position
                    let origin_x = place_x - g_min[0];
                    let origin_y = place_y - g_min[1];

                    // Compute valid position bounds and clamp to ensure geometry stays within boundary
                    // Ensure origin position >= boundary min for multi-strip compatibility
                    let (g_min_rot, g_max_rot) = geom.aabb_at_rotation(rotation);
                    let min_valid_x = (b_min[0] + margin - g_min_rot[0]).max(b_min[0] + margin);
                    let max_valid_x = b_max[0] - margin - g_max_rot[0];
                    let min_valid_y = (b_min[1] + margin - g_min_rot[1]).max(b_min[1] + margin);
                    let max_valid_y = b_max[1] - margin - g_max_rot[1];

                    let clamped_x = origin_x.clamp(min_valid_x, max_valid_x);
                    let clamped_y = origin_y.clamp(min_valid_y, max_valid_y);

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        instance,
                        clamped_x,
                        clamped_y,
                        rotation,
                    );

                    placements.push(placement);
                    total_placed_area += geom.measure();
                    placed_count += 1;

                    current_x = place_x + g_width + spacing;
                    current_y = place_y;
                    row_height = row_height.max(g_height);

                    // Progress callback every piece
                    callback(
                        ProgressInfo::new()
                            .with_phase("BLF Placement")
                            .with_items(placed_count, total_pieces)
                            .with_utilization(total_placed_area / boundary.measure())
                            .with_elapsed(start.elapsed().as_millis() as u64),
                    );
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
        callback(
            ProgressInfo::new()
                .with_phase("Complete")
                .with_items(placed_count, total_pieces)
                .with_utilization(result.utilization)
                .with_elapsed(result.computation_time_ms)
                .finished(),
        );

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
        callback(
            ProgressInfo::new()
                .with_phase("NFP Placement")
                .with_items(0, total_pieces)
                .with_elapsed(0),
        );

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
                    callback(
                        ProgressInfo::new()
                            .with_phase("Cancelled")
                            .with_items(placed_count, total_pieces)
                            .with_elapsed(result.computation_time_ms)
                            .finished(),
                    );
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
                    callback(
                        ProgressInfo::new()
                            .with_phase("Time Limit Reached")
                            .with_items(placed_count, total_pieces)
                            .with_elapsed(result.computation_time_ms)
                            .finished(),
                    );
                    return Ok(result);
                }

                let mut best_placement: Option<(f64, f64, f64)> = None;

                for &rotation in &rotation_angles {
                    let ifp =
                        match compute_ifp_with_margin(&boundary_polygon, geom, rotation, margin) {
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

                        // Compute NFP at origin and cache it (with relative rotation)
                        // Formula: NFP_actual = translate(rotate(NFP_relative, placed.rotation), placed.position)
                        let nfp_at_origin = match self.nfp_cache.get_or_compute(cache_key, || {
                            let placed_at_origin = placed.geometry.clone();
                            compute_nfp(&placed_at_origin, geom, rotation - placed.rotation)
                        }) {
                            Ok(nfp) => nfp,
                            Err(_) => continue,
                        };

                        // Transform NFP: first rotate by placed.rotation, then translate
                        let rotated_nfp = rotate_nfp(&nfp_at_origin, placed.rotation);
                        let translated_nfp = translate_nfp(&rotated_nfp, placed.position);
                        let expanded = self.expand_nfp(&translated_nfp, spacing);
                        nfps.push(expanded);
                    }

                    let ifp_shrunk = self.shrink_ifp(&ifp, spacing);
                    let nfp_refs: Vec<&Nfp> = nfps.iter().collect();

                    if let Some((x, y)) =
                        find_bottom_left_placement(&ifp_shrunk, &nfp_refs, sample_step)
                    {
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
                    // Compute valid position bounds and clamp to ensure geometry stays within boundary
                    // Ensure origin position >= boundary min for multi-strip compatibility
                    let (b_min, b_max) = boundary.aabb();
                    let (g_min, g_max) = geom.aabb_at_rotation(rotation);
                    let min_valid_x = (b_min[0] + margin - g_min[0]).max(b_min[0] + margin);
                    let max_valid_x = b_max[0] - margin - g_max[0];
                    let min_valid_y = (b_min[1] + margin - g_min[1]).max(b_min[1] + margin);
                    let max_valid_y = b_max[1] - margin - g_max[1];

                    let clamped_x = x.clamp(min_valid_x, max_valid_x);
                    let clamped_y = y.clamp(min_valid_y, max_valid_y);

                    let placement = Placement::new_2d(
                        geom.id().clone(),
                        instance,
                        clamped_x,
                        clamped_y,
                        rotation,
                    );
                    placements.push(placement);
                    placed_geometries.push(PlacedGeometry::new(
                        geom.clone(),
                        (clamped_x, clamped_y),
                        rotation,
                    ));
                    total_placed_area += geom.measure();
                    placed_count += 1;

                    // Progress callback every piece
                    callback(
                        ProgressInfo::new()
                            .with_phase("NFP Placement")
                            .with_items(placed_count, total_pieces)
                            .with_utilization(total_placed_area / boundary.measure())
                            .with_elapsed(start.elapsed().as_millis() as u64),
                    );
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
        callback(
            ProgressInfo::new()
                .with_phase("Complete")
                .with_items(placed_count, total_pieces)
                .with_utilization(result.utilization)
                .with_elapsed(result.computation_time_ms)
                .finished(),
        );

        Ok(result)
    }

    /// Solves nesting with automatic multi-strip support.
    ///
    /// When items don't fit in a single strip, automatically creates additional strips.
    /// Each placement's `boundary_index` indicates which strip it belongs to.
    /// Positions are adjusted so that strip N items have x offset of N * strip_width.
    pub fn solve_multi_strip(
        &self,
        geometries: &[Geometry2D],
        boundary: &Boundary2D,
    ) -> Result<SolveResult<f64>> {
        boundary.validate()?;
        self.cancelled.store(false, Ordering::Relaxed);

        let (b_min, b_max) = boundary.aabb();
        let strip_width = b_max[0] - b_min[0];

        let mut final_result = SolveResult::new();
        let mut remaining_geometries: Vec<Geometry2D> = geometries.to_vec();
        let mut strip_index = 0;
        let max_strips = 100; // Safety limit

        while !remaining_geometries.is_empty() && strip_index < max_strips {
            if self.cancelled.load(Ordering::Relaxed) {
                break;
            }

            // Solve on current strip
            let strip_result = match self.config.strategy {
                Strategy::BottomLeftFill => self.bottom_left_fill(&remaining_geometries, boundary),
                Strategy::NfpGuided => self.nfp_guided_blf(&remaining_geometries, boundary),
                Strategy::GeneticAlgorithm => {
                    self.genetic_algorithm(&remaining_geometries, boundary)
                }
                Strategy::Brkga => self.brkga(&remaining_geometries, boundary),
                Strategy::SimulatedAnnealing => {
                    self.simulated_annealing(&remaining_geometries, boundary)
                }
                Strategy::Gdrr => self.gdrr(&remaining_geometries, boundary),
                Strategy::Alns => self.alns(&remaining_geometries, boundary),
                #[cfg(feature = "milp")]
                Strategy::MilpExact => self.milp_exact(&remaining_geometries, boundary),
                #[cfg(feature = "milp")]
                Strategy::HybridExact => self.hybrid_exact(&remaining_geometries, boundary),
                _ => self.nfp_guided_blf(&remaining_geometries, boundary),
            }?;

            if strip_result.placements.is_empty() {
                // No progress - items too large for strip
                final_result.unplaced.extend(strip_result.unplaced);
                break;
            }

            // Collect placed geometry IDs
            let placed_ids: std::collections::HashSet<_> = strip_result
                .placements
                .iter()
                .map(|p| p.geometry_id.clone())
                .collect();

            // Adjust placements for this strip and add to final result
            for mut placement in strip_result.placements {
                // Offset x position by strip_index * strip_width
                if !placement.position.is_empty() {
                    placement.position[0] += strip_index as f64 * strip_width;
                }
                placement.boundary_index = strip_index;
                final_result.placements.push(placement);
            }

            // Update remaining geometries (those not placed)
            remaining_geometries.retain(|g| !placed_ids.contains(g.id()));

            // Also handle quantity > 1: reduce quantity for partially placed items
            // For now, we treat each geometry independently

            strip_index += 1;
        }

        final_result.boundaries_used = strip_index;
        final_result.deduplicate_unplaced();

        // Calculate overall utilization
        let total_boundary_area = boundary.measure() * strip_index as f64;
        if total_boundary_area > 0.0 {
            let placed_area: f64 = final_result
                .placements
                .iter()
                .filter_map(|p| {
                    geometries
                        .iter()
                        .find(|g| g.id() == &p.geometry_id)
                        .map(|g| {
                            use u_nesting_core::geometry::Geometry;
                            g.measure()
                        })
                })
                .sum();
            final_result.utilization = placed_area / total_boundary_area;
        }

        Ok(final_result)
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
            Strategy::Gdrr => self.gdrr(geometries, boundary),
            Strategy::Alns => self.alns(geometries, boundary),
            #[cfg(feature = "milp")]
            Strategy::MilpExact => self.milp_exact(geometries, boundary),
            #[cfg(feature = "milp")]
            Strategy::HybridExact => self.hybrid_exact(geometries, boundary),
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
                let mut ga_config = GaConfig::default()
                    .with_population_size(self.config.population_size)
                    .with_max_generations(self.config.max_generations)
                    .with_crossover_rate(self.config.crossover_rate)
                    .with_mutation_rate(self.config.mutation_rate);

                // Apply time limit if specified
                if self.config.time_limit_ms > 0 {
                    ga_config = ga_config.with_time_limit(std::time::Duration::from_millis(
                        self.config.time_limit_ms,
                    ));
                }

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
    fn test_gdrr_strategy_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default().with_strategy(Strategy::Gdrr);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        // GDRR should report iterations and fitness
        assert!(result.iterations.is_some());
        assert!(result.best_fitness.is_some());
        assert!(result.strategy == Some("GDRR".to_string()));
    }

    #[test]
    fn test_gdrr_strategy_all_placed() {
        // Easy case: 4 small rectangles in large boundary
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::Gdrr);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // All 4 pieces should fit
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_alns_strategy_basic() {
        let geometries = vec![
            Geometry2D::rectangle("R1", 20.0, 10.0).with_quantity(2),
            Geometry2D::rectangle("R2", 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary2D::rectangle(100.0, 50.0);
        let config = Config::default().with_strategy(Strategy::Alns);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        // ALNS should report iterations and fitness
        assert!(result.iterations.is_some());
        assert!(result.best_fitness.is_some());
        assert!(result.strategy == Some("ALNS".to_string()));
    }

    #[test]
    fn test_alns_strategy_all_placed() {
        // Easy case: 4 small rectangles in large boundary
        let geometries = vec![Geometry2D::rectangle("R1", 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::Alns);
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
        let geometries = vec![Geometry2D::rectangle("R1", 30.0, 10.0)
                .with_rotations(vec![0.0, std::f64::consts::FRAC_PI_2]) // 0 and 90 degrees
                .with_quantity(3)];

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
        let geometries = vec![Geometry2D::rectangle("R1", 40.0, 10.0)
                .with_rotations(vec![0.0, std::f64::consts::FRAC_PI_2]) // 0 and 90 degrees
                .with_quantity(2)];

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

        let result = nester
            .solve_with_progress(&geometries, &boundary, callback)
            .unwrap();

        // Verify callback was called (at least once per piece + initial + final)
        let count = callback_count.load(Ordering::Relaxed);
        assert!(
            count >= 5,
            "Expected at least 5 callbacks (1 initial + 4 pieces + 1 final), got {}",
            count
        );

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

        let result = nester
            .solve_with_progress(&geometries, &boundary, callback)
            .unwrap();

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

    #[test]
    fn test_blf_bounds_clamping() {
        // Test that BLF correctly clamps placements within boundary
        // Create a shape with non-zero g_min (similar to Gear shape)
        // Gear-like: x ranges from 5 to 95 (width=90), y from 5 to 95 (height=90)
        let gear_like = Geometry2D::new("gear")
            .with_polygon(vec![
                (50.0, 5.0), // Bottom
                (65.0, 15.0),
                (77.0, 18.0),
                (80.0, 32.0),
                (95.0, 50.0), // Right
                (80.0, 68.0),
                (77.0, 82.0),
                (65.0, 85.0),
                (50.0, 95.0), // Top
                (35.0, 85.0),
                (23.0, 82.0),
                (20.0, 68.0),
                (5.0, 50.0), // Left (min_x = 5)
                (20.0, 32.0),
                (23.0, 18.0),
                (35.0, 15.0),
            ])
            .with_quantity(1);

        // Boundary is 100x100
        let boundary = Boundary2D::rectangle(100.0, 100.0);

        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let result = nester.solve(&[gear_like.clone()], &boundary).unwrap();

        assert_eq!(result.placements.len(), 1);
        let placement = &result.placements[0];

        // Origin position
        let origin_x = placement.position[0];
        let origin_y = placement.position[1];

        // Get rotation from placement (2D rotation is a single value in Vec)
        let rotation = placement.rotation.first().copied().unwrap_or(0.0);

        // Get AABB at rotation
        let (g_min, g_max) = gear_like.aabb_at_rotation(rotation);

        // Actual geometry bounds after placement
        let actual_min_x = origin_x + g_min[0];
        let actual_max_x = origin_x + g_max[0];
        let actual_min_y = origin_y + g_min[1];
        let actual_max_y = origin_y + g_max[1];

        // All edges should be within boundary [0, 100]
        assert!(
            actual_min_x >= 0.0,
            "Left edge {} should be >= 0",
            actual_min_x
        );
        assert!(
            actual_max_x <= 100.0,
            "Right edge {} should be <= 100",
            actual_max_x
        );
        assert!(
            actual_min_y >= 0.0,
            "Bottom edge {} should be >= 0",
            actual_min_y
        );
        assert!(
            actual_max_y <= 100.0,
            "Top edge {} should be <= 100",
            actual_max_y
        );
    }

    #[test]
    fn test_blf_bounds_clamping_many_pieces() {
        // Test BLF bounds clamping with many pieces to trigger row overflow
        // This mimics the actual failing case from test_blf.py
        let gear_like = Geometry2D::new("gear")
            .with_polygon(vec![
                (50.0, 5.0),
                (65.0, 15.0),
                (77.0, 18.0),
                (80.0, 32.0),
                (95.0, 50.0),
                (80.0, 68.0),
                (77.0, 82.0),
                (65.0, 85.0),
                (50.0, 95.0),
                (35.0, 85.0),
                (23.0, 82.0),
                (20.0, 68.0),
                (5.0, 50.0),
                (20.0, 32.0),
                (23.0, 18.0),
                (35.0, 15.0),
            ])
            .with_quantity(13); // Same as Gear (shape 8) in test_blf.py

        // Boundary is 500x500 like the test
        let boundary = Boundary2D::rectangle(500.0, 500.0);

        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let result = nester.solve(&[gear_like.clone()], &boundary).unwrap();

        // Check that ALL placements are within bounds
        for (i, placement) in result.placements.iter().enumerate() {
            let origin_x = placement.position[0];
            let origin_y = placement.position[1];
            let rotation = placement.rotation.first().copied().unwrap_or(0.0);

            let (g_min, g_max) = gear_like.aabb_at_rotation(rotation);

            let actual_min_x = origin_x + g_min[0];
            let actual_max_x = origin_x + g_max[0];
            let actual_min_y = origin_y + g_min[1];
            let actual_max_y = origin_y + g_max[1];

            assert!(
                actual_min_x >= -0.01,
                "Piece {}: Left edge {} should be >= 0",
                i,
                actual_min_x
            );
            assert!(
                actual_max_x <= 500.01,
                "Piece {}: Right edge {} should be <= 500",
                i,
                actual_max_x
            );
            assert!(
                actual_min_y >= -0.01,
                "Piece {}: Bottom edge {} should be >= 0",
                i,
                actual_min_y
            );
            assert!(
                actual_max_y <= 500.01,
                "Piece {}: Top edge {} should be <= 500",
                i,
                actual_max_y
            );
        }
    }

    #[test]
    fn test_blf_bounds_trace() {
        // Debug test: trace through BLF to understand why clamping doesn't work
        let gear = Geometry2D::new("gear").with_polygon(vec![
            (50.0, 5.0),
            (65.0, 15.0),
            (77.0, 18.0),
            (80.0, 32.0),
            (95.0, 50.0),
            (80.0, 68.0),
            (77.0, 82.0),
            (65.0, 85.0),
            (50.0, 95.0),
            (35.0, 85.0),
            (23.0, 82.0),
            (20.0, 68.0),
            (5.0, 50.0),
            (20.0, 32.0),
            (23.0, 18.0),
            (35.0, 15.0),
        ]);

        // Verify AABB
        let (g_min, g_max) = gear.aabb();
        println!("Gear AABB: min={:?}, max={:?}", g_min, g_max);
        assert!(
            (g_min[0] - 5.0).abs() < 0.01,
            "g_min[0] should be 5, got {}",
            g_min[0]
        );
        assert!(
            (g_max[0] - 95.0).abs() < 0.01,
            "g_max[0] should be 95, got {}",
            g_max[0]
        );

        // Verify valid origin range for 500x500 boundary
        let b_max_x = 500.0;
        let margin = 0.0;
        let max_valid_x = b_max_x - margin - g_max[0];
        println!(
            "max_valid_x = {} - {} - {} = {}",
            b_max_x, margin, g_max[0], max_valid_x
        );
        assert!(
            (max_valid_x - 405.0).abs() < 0.01,
            "max_valid_x should be 405, got {}",
            max_valid_x
        );

        // Run BLF and check the result
        let boundary = Boundary2D::rectangle(500.0, 500.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let result = nester
            .solve(&[gear.clone().with_quantity(1)], &boundary)
            .unwrap();

        assert_eq!(result.placements.len(), 1);
        let p = &result.placements[0];
        let origin_x = p.position[0];
        let rotation = p.rotation.first().copied().unwrap_or(0.0);

        let (g_min_r, g_max_r) = gear.aabb_at_rotation(rotation);
        let actual_max_x = origin_x + g_max_r[0];

        println!("Placement: origin_x={}, rotation={}", origin_x, rotation);
        println!(
            "At rotation {}: g_min={:?}, g_max={:?}",
            rotation, g_min_r, g_max_r
        );
        println!(
            "Actual max x: {} + {} = {}",
            origin_x, g_max_r[0], actual_max_x
        );

        assert!(
            actual_max_x <= 500.01,
            "Geometry exceeds boundary: max_x={} > 500",
            actual_max_x
        );
    }

    #[test]
    fn test_blf_bounds_many_pieces_direct() {
        // Test with many pieces to trigger the boundary violation
        let gear = Geometry2D::new("gear")
            .with_polygon(vec![
                (50.0, 5.0),
                (65.0, 15.0),
                (77.0, 18.0),
                (80.0, 32.0),
                (95.0, 50.0),
                (80.0, 68.0),
                (77.0, 82.0),
                (65.0, 85.0),
                (50.0, 95.0),
                (35.0, 85.0),
                (23.0, 82.0),
                (20.0, 68.0),
                (5.0, 50.0),
                (20.0, 32.0),
                (23.0, 18.0),
                (35.0, 15.0),
            ])
            .with_quantity(25); // Many pieces

        let boundary = Boundary2D::rectangle(500.0, 500.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let result = nester.solve(&[gear.clone()], &boundary).unwrap();

        println!("Placed {} pieces", result.placements.len());

        // Check all placements
        for (i, p) in result.placements.iter().enumerate() {
            let origin_x = p.position[0];
            let origin_y = p.position[1];
            let rotation = p.rotation.first().copied().unwrap_or(0.0);

            let (g_min_r, g_max_r) = gear.aabb_at_rotation(rotation);

            let actual_min_x = origin_x + g_min_r[0];
            let actual_max_x = origin_x + g_max_r[0];
            let actual_min_y = origin_y + g_min_r[1];
            let actual_max_y = origin_y + g_max_r[1];

            println!(
                "Piece {}: origin=({:.1}, {:.1}), rot={:.2}, bounds=[{:.1},{:.1}]x[{:.1},{:.1}]",
                i,
                origin_x,
                origin_y,
                rotation,
                actual_min_x,
                actual_max_x,
                actual_min_y,
                actual_max_y
            );

            assert!(
                actual_max_x <= 500.01,
                "Piece {}: Right edge {} > 500",
                i,
                actual_max_x
            );
            assert!(
                actual_max_y <= 500.01,
                "Piece {}: Top edge {} > 500",
                i,
                actual_max_y
            );
        }
    }

    #[test]
    fn test_blf_bounds_multi_strip() {
        // Test with solve_multi_strip which is what benchmark runner uses
        let gear = Geometry2D::new("gear")
            .with_polygon(vec![
                (50.0, 5.0),
                (65.0, 15.0),
                (77.0, 18.0),
                (80.0, 32.0),
                (95.0, 50.0),
                (80.0, 68.0),
                (77.0, 82.0),
                (65.0, 85.0),
                (50.0, 95.0),
                (35.0, 85.0),
                (23.0, 82.0),
                (20.0, 68.0),
                (5.0, 50.0),
                (20.0, 32.0),
                (23.0, 18.0),
                (35.0, 15.0),
            ])
            .with_quantity(50); // Many pieces to force multiple strips

        let boundary = Boundary2D::rectangle(500.0, 500.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        // Use solve_multi_strip like benchmark runner does
        let result = nester
            .solve_multi_strip(&[gear.clone()], &boundary)
            .unwrap();

        println!(
            "Placed {} pieces across {} strips",
            result.placements.len(),
            result.boundaries_used
        );

        // Check all placements - within their respective strips
        let strip_width = 500.0;
        for (i, p) in result.placements.iter().enumerate() {
            let origin_x = p.position[0];
            let origin_y = p.position[1];
            let rotation = p.rotation.first().copied().unwrap_or(0.0);
            let strip_idx = p.boundary_index;

            // Calculate local position within strip
            let local_x = origin_x - (strip_idx as f64 * strip_width);

            let (_g_min_r, g_max_r) = gear.aabb_at_rotation(rotation);

            let local_max_x = local_x + g_max_r[0];
            let local_max_y = origin_y + g_max_r[1];

            println!(
                "Piece {}: strip={}, origin=({:.1}, {:.1}), local_x={:.1}, rot={:.2}, local_max_x={:.1}",
                i, strip_idx, origin_x, origin_y, local_x, rotation, local_max_x
            );

            assert!(
                local_max_x <= 500.01,
                "Piece {}: In strip {}, local right edge {:.1} > 500",
                i,
                strip_idx,
                local_max_x
            );
            assert!(
                local_max_y <= 500.01,
                "Piece {}: Top edge {:.1} > 500",
                i,
                local_max_y
            );
        }
    }

    #[test]
    fn test_blf_bounds_mixed_shapes() {
        // Replicate test_blf.py with all 9 shapes
        let shapes = vec![
            // Shape 0: Rounded rectangle (demand 2)
            Geometry2D::new("shape0")
                .with_polygon(vec![
                    (0.0, 0.0),
                    (180.0, 0.0),
                    (195.0, 15.0),
                    (200.0, 50.0),
                    (200.0, 150.0),
                    (195.0, 185.0),
                    (180.0, 200.0),
                    (20.0, 200.0),
                    (5.0, 185.0),
                    (0.0, 150.0),
                    (0.0, 50.0),
                    (5.0, 15.0),
                ])
                .with_quantity(2),
            // Shape 1: Circular-ish (demand 4)
            Geometry2D::new("shape1")
                .with_polygon(vec![
                    (60.0, 0.0),
                    (85.0, 7.0),
                    (104.0, 25.0),
                    (118.0, 50.0),
                    (120.0, 60.0),
                    (118.0, 70.0),
                    (104.0, 95.0),
                    (85.0, 113.0),
                    (60.0, 120.0),
                    (35.0, 113.0),
                    (16.0, 95.0),
                    (2.0, 70.0),
                    (0.0, 60.0),
                    (2.0, 50.0),
                    (16.0, 25.0),
                    (35.0, 7.0),
                ])
                .with_quantity(4),
            // Shape 2: L-shape (demand 6)
            Geometry2D::new("shape2")
                .with_polygon(vec![
                    (0.0, 0.0),
                    (80.0, 0.0),
                    (80.0, 20.0),
                    (20.0, 20.0),
                    (20.0, 80.0),
                    (0.0, 80.0),
                ])
                .with_quantity(6),
            // Shape 3: Triangle (demand 6)
            Geometry2D::new("shape3")
                .with_polygon(vec![(0.0, 0.0), (70.0, 0.0), (0.0, 70.0)])
                .with_quantity(6),
            // Shape 4: Rectangle (demand 4)
            Geometry2D::new("shape4")
                .with_polygon(vec![(0.0, 0.0), (120.0, 0.0), (120.0, 60.0), (0.0, 60.0)])
                .with_quantity(4),
            // Shape 5: Hexagon (demand 8)
            Geometry2D::new("shape5")
                .with_polygon(vec![
                    (15.0, 0.0),
                    (45.0, 0.0),
                    (60.0, 26.0),
                    (45.0, 52.0),
                    (15.0, 52.0),
                    (0.0, 26.0),
                ])
                .with_quantity(8),
            // Shape 6: T-shape (demand 4)
            Geometry2D::new("shape6")
                .with_polygon(vec![
                    (0.0, 0.0),
                    (90.0, 0.0),
                    (90.0, 12.0),
                    (55.0, 12.0),
                    (55.0, 60.0),
                    (35.0, 60.0),
                    (35.0, 12.0),
                    (0.0, 12.0),
                ])
                .with_quantity(4),
            // Shape 7: Rounded square (demand 3)
            Geometry2D::new("shape7")
                .with_polygon(vec![
                    (0.0, 10.0),
                    (10.0, 0.0),
                    (70.0, 0.0),
                    (80.0, 10.0),
                    (80.0, 70.0),
                    (70.0, 80.0),
                    (10.0, 80.0),
                    (0.0, 70.0),
                ])
                .with_quantity(3),
            // Shape 8: Gear (demand 13) - the problematic shape
            Geometry2D::new("shape8_gear")
                .with_polygon(vec![
                    (50.0, 5.0),
                    (65.0, 15.0),
                    (77.0, 18.0),
                    (80.0, 32.0),
                    (95.0, 50.0),
                    (80.0, 68.0),
                    (77.0, 82.0),
                    (65.0, 85.0),
                    (50.0, 95.0),
                    (35.0, 85.0),
                    (23.0, 82.0),
                    (20.0, 68.0),
                    (5.0, 50.0),
                    (20.0, 32.0),
                    (23.0, 18.0),
                    (35.0, 15.0),
                ])
                .with_quantity(13),
        ];

        // Total: 2+4+6+6+4+8+4+3+13 = 50 pieces
        let boundary = Boundary2D::rectangle(500.0, 500.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let result = nester.solve_multi_strip(&shapes, &boundary).unwrap();

        println!(
            "Placed {} pieces across {} strips",
            result.placements.len(),
            result.boundaries_used
        );

        // Check placements for Gear (shape8) specifically
        let strip_width = 500.0;
        let gear_aabb = shapes[8].aabb();
        println!("Gear AABB: min={:?}, max={:?}", gear_aabb.0, gear_aabb.1);

        let mut violations = Vec::new();
        for p in &result.placements {
            if p.geometry_id.as_str().starts_with("shape8") {
                let origin_x = p.position[0];
                let _origin_y = p.position[1];
                let rotation = p.rotation.first().copied().unwrap_or(0.0);
                let strip_idx = p.boundary_index;
                let local_x = origin_x - (strip_idx as f64 * strip_width);

                let (_g_min_r, g_max_r) = shapes[8].aabb_at_rotation(rotation);
                let local_max_x = local_x + g_max_r[0];

                println!(
                    "{}: strip={}, local_x={:.1}, rot={:.2}, local_max_x={:.1}",
                    p.geometry_id, strip_idx, local_x, rotation, local_max_x
                );

                if local_max_x > 500.01 {
                    violations.push((p.geometry_id.clone(), strip_idx, local_x, local_max_x));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "Found {} Gear pieces exceeding boundary: {:?}",
            violations.len(),
            violations
        );
    }

    #[test]
    fn test_blf_bounds_expanded_like_benchmark() {
        // Replicate EXACTLY how benchmark runner creates geometries:
        // Each piece is a separate Geometry2D with quantity=1
        // (vertices, demand, allowed_rotations_deg)
        let shape_defs: Vec<(Vec<(f64, f64)>, usize, Vec<f64>)> = vec![
            (
                vec![
                    (0.0, 0.0),
                    (180.0, 0.0),
                    (195.0, 15.0),
                    (200.0, 50.0),
                    (200.0, 150.0),
                    (195.0, 185.0),
                    (180.0, 200.0),
                    (20.0, 200.0),
                    (5.0, 185.0),
                    (0.0, 150.0),
                    (0.0, 50.0),
                    (5.0, 15.0),
                ],
                2,
                vec![0.0, 90.0, 180.0, 270.0],
            ),
            (
                vec![
                    (60.0, 0.0),
                    (85.0, 7.0),
                    (104.0, 25.0),
                    (118.0, 50.0),
                    (120.0, 60.0),
                    (118.0, 70.0),
                    (104.0, 95.0),
                    (85.0, 113.0),
                    (60.0, 120.0),
                    (35.0, 113.0),
                    (16.0, 95.0),
                    (2.0, 70.0),
                    (0.0, 60.0),
                    (2.0, 50.0),
                    (16.0, 25.0),
                    (35.0, 7.0),
                ],
                4,
                vec![0.0, 45.0, 90.0, 135.0],
            ),
            (
                vec![
                    (0.0, 0.0),
                    (80.0, 0.0),
                    (80.0, 20.0),
                    (20.0, 20.0),
                    (20.0, 80.0),
                    (0.0, 80.0),
                ],
                6,
                vec![0.0, 90.0, 180.0, 270.0],
            ),
            (
                vec![(0.0, 0.0), (70.0, 0.0), (0.0, 70.0)],
                6,
                vec![0.0, 90.0, 180.0, 270.0],
            ),
            (
                vec![(0.0, 0.0), (120.0, 0.0), (120.0, 60.0), (0.0, 60.0)],
                4,
                vec![0.0, 90.0],
            ),
            (
                vec![
                    (15.0, 0.0),
                    (45.0, 0.0),
                    (60.0, 26.0),
                    (45.0, 52.0),
                    (15.0, 52.0),
                    (0.0, 26.0),
                ],
                8,
                vec![0.0, 60.0, 120.0],
            ),
            (
                vec![
                    (0.0, 0.0),
                    (90.0, 0.0),
                    (90.0, 12.0),
                    (55.0, 12.0),
                    (55.0, 60.0),
                    (35.0, 60.0),
                    (35.0, 12.0),
                    (0.0, 12.0),
                ],
                4,
                vec![0.0, 90.0, 180.0, 270.0],
            ),
            (
                vec![
                    (0.0, 10.0),
                    (10.0, 0.0),
                    (70.0, 0.0),
                    (80.0, 10.0),
                    (80.0, 70.0),
                    (70.0, 80.0),
                    (10.0, 80.0),
                    (0.0, 70.0),
                ],
                3,
                vec![0.0, 90.0],
            ),
            // Shape 8: Gear - with all 8 rotations
            (
                vec![
                    (50.0, 5.0),
                    (65.0, 15.0),
                    (77.0, 18.0),
                    (80.0, 32.0),
                    (95.0, 50.0),
                    (80.0, 68.0),
                    (77.0, 82.0),
                    (65.0, 85.0),
                    (50.0, 95.0),
                    (35.0, 85.0),
                    (23.0, 82.0),
                    (20.0, 68.0),
                    (5.0, 50.0),
                    (20.0, 32.0),
                    (23.0, 18.0),
                    (35.0, 15.0),
                ],
                13,
                vec![0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
            ),
        ];

        // Expand like benchmark runner: each piece is separate geometry
        let mut geometries = Vec::new();
        let mut piece_id = 0;
        for (vertices, demand, rotations) in shape_defs.iter() {
            for _ in 0..*demand {
                let geom = Geometry2D::new(format!("piece_{}", piece_id))
                    .with_polygon(vertices.clone())
                    .with_rotations_deg(rotations.clone());
                geometries.push(geom);
                piece_id += 1;
            }
        }

        // Store gear AABB for checking
        let gear_geom = Geometry2D::new("gear_check").with_polygon(shape_defs[8].0.clone());
        let (gear_min, gear_max) = gear_geom.aabb();
        println!("Gear AABB: min={:?}, max={:?}", gear_min, gear_max);

        let boundary = Boundary2D::rectangle(500.0, 500.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let nester = Nester2D::new(config);

        let result = nester.solve_multi_strip(&geometries, &boundary).unwrap();

        println!(
            "Placed {} pieces across {} strips",
            result.placements.len(),
            result.boundaries_used
        );

        // Check Gear placements (piece_37 to piece_49)
        let strip_width = 500.0;
        let mut violations = Vec::new();

        for p in &result.placements {
            let id_num: usize = p
                .geometry_id
                .as_str()
                .strip_prefix("piece_")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            // piece_37 to piece_49 are Gear shapes
            if id_num >= 37 && id_num <= 49 {
                let origin_x = p.position[0];
                let rotation = p.rotation.first().copied().unwrap_or(0.0);
                let strip_idx = p.boundary_index;
                let local_x = origin_x - (strip_idx as f64 * strip_width);

                let (_, g_max_r) = gear_geom.aabb_at_rotation(rotation);
                let local_max_x = local_x + g_max_r[0];

                println!(
                    "{}: strip={}, local_x={:.1}, rot={:.2}, local_max_x={:.1}",
                    p.geometry_id, strip_idx, local_x, rotation, local_max_x
                );

                if local_max_x > 500.01 {
                    violations.push((p.geometry_id.clone(), strip_idx, local_x, local_max_x));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "Found {} Gear pieces exceeding boundary: {:?}",
            violations.len(),
            violations
        );
    }
}
