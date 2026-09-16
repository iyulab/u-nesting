//! NFP Covering Model (NFP-CM) MILP solver for 2D nesting.
//!
//! This module implements an exact solver using the NFP Covering Model approach,
//! which uses No-Fit Polygons to define valid placement regions and formulates
//! the problem as a Mixed Integer Linear Program.
//!
//! # Algorithm
//!
//! The NFP-CM approach (Lastra-Díaz & Ortuño, 2023):
//! 1. Precompute NFPs between all piece pairs
//! 2. Discretize the boundary into candidate placement points
//! 3. Binary variable for each (piece, position, rotation) triple
//! 4. Non-overlap constraints derived from NFP geometry
//! 5. Objective: minimize strip length
//!
//! # Advantages over Basic MILP
//!
//! - Tighter formulation for irregular pieces
//! - Better LP relaxation bounds
//! - Handles concave pieces naturally
//!
//! # Limitations
//!
//! - Requires NFP precomputation (can be expensive)
//! - Grid discretization limits solution precision
//! - Still NP-hard, only suitable for small instances (≤15-20 pieces)
//!
//! # References
//!
//! - Lastra-Díaz, J. J., & Ortuño, M. T. (2023). "NFP-CM: A MILP formulation for
//!   the irregular strip packing problem based on No-Fit Polygons"

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
#[cfg(feature = "milp")]
use crate::nfp::{compute_nfp_mirrored, rotate_nfp, translate_nfp, Nfp};
use crate::placement_utils::offset_nfp;
#[cfg(feature = "milp")]
use u_nesting_core::exact::{ExactConfig, ExactResult};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

#[cfg(feature = "milp")]
use good_lp::{
    constraint, default_solver, variable, Expression, ProblemVariables, Solution, SolverModel,
    Variable, WithTimeLimit,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Candidate placement position.
#[derive(Debug, Clone)]
struct CandidatePosition {
    /// X coordinate — grid anchor of the piece's *AABB* at this rotation,
    /// used for the strip-length constraint (which is stated in terms of
    /// AABB width/height). NOT the `Placement` origin — see `origin_x`.
    x: f64,
    /// Y coordinate — see `x`.
    y: f64,
    /// The `Placement`/NFP-frame origin translation for this candidate,
    /// i.e. `x` corrected by the piece's own rotated (and mirrored) local
    /// AABB minimum (`x - g_min[0]`). This is the coordinate space every
    /// other strategy places pieces in, and the one `compute_nfp_mirrored`'s
    /// output assumes for whichever geometry is passed as `stationary`.
    /// Precomputed once per (rotation, mirror) here instead of recomputing
    /// per grid cell or per conflict-candidate pair.
    origin_x: f64,
    /// See `origin_x`.
    origin_y: f64,
    /// Rotation angle in radians.
    rotation: f64,
    /// Rotation index.
    rotation_idx: usize,
    /// Whether this candidate is mirrored (`allow_flip` support).
    mirror: bool,
}

/// Piece info with candidate positions.
#[derive(Debug, Clone)]
struct PieceInfo {
    /// Original geometry index.
    geometry_idx: usize,
    /// Instance number.
    instance_num: usize,
    /// Geometry ID.
    id: String,
    /// Area.
    area: f64,
    /// Width at each rotation.
    widths: Vec<f64>,
    /// Height at each rotation.
    heights: Vec<f64>,
    /// Candidate positions (x, y, rotation_idx).
    candidates: Vec<CandidatePosition>,
}

/// NFP-CM solution.
#[derive(Debug, Clone)]
struct NfpCmSolution {
    /// Placements: (piece_idx, candidate_idx).
    assignments: Vec<(usize, usize)>,
    /// Objective value.
    objective: f64,
    /// Exact result info.
    exact_result: ExactResult,
}

/// Run NFP-CM MILP solver.
///
/// This formulation uses NFPs to define valid placement regions and selects
/// from discretized candidate positions.
#[cfg(feature = "milp")]
pub fn run_nfp_cm_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    exact_config: &ExactConfig,
    cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    let start = Instant::now();
    let mut result = SolveResult::new();

    // Count total instances
    let total_instances: usize = geometries.iter().map(|g| g.quantity()).sum();

    if !exact_config.is_within_limit(total_instances) {
        log::warn!(
            "Instance count {} exceeds exact limit {}",
            total_instances,
            exact_config.max_items
        );
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        return result;
    }

    // Get boundary dimensions
    let (b_min, b_max) = boundary.aabb();
    let margin = config.margin;
    let bound_width = b_max[0] - b_min[0] - 2.0 * margin;
    let bound_height = b_max[1] - b_min[1] - 2.0 * margin;

    if bound_width <= 0.0 || bound_height <= 0.0 {
        log::error!("Invalid boundary dimensions");
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        return result;
    }

    let rotation_angles = exact_config.rotation_angles();
    let grid_step = exact_config.grid_step;

    // Build piece info with candidate positions
    let pieces = build_piece_info(
        geometries,
        boundary,
        config,
        &rotation_angles,
        grid_step,
        &cancelled,
    );

    if pieces.is_empty() {
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        return result;
    }

    // Precompute NFP conflicts between candidates
    let conflicts = compute_conflicts(
        &pieces,
        geometries,
        config.spacing,
        &cancelled,
        start,
        exact_config.time_limit_ms,
    );

    if cancelled.load(Ordering::Relaxed) {
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        return result;
    }

    // Solve NFP-CM MILP
    match solve_nfp_cm_milp(
        &pieces,
        &conflicts,
        bound_width,
        b_min[0] + margin,
        b_min[1] + margin,
        &rotation_angles,
        exact_config,
        &cancelled,
        start,
    ) {
        Some(solution) => {
            // Convert solution to placements
            for (piece_idx, candidate_idx) in &solution.assignments {
                let piece = &pieces[*piece_idx];
                let candidate = &piece.candidates[*candidate_idx];

                result.placements.push(
                    Placement::new_2d(
                        piece.id.clone(),
                        piece.instance_num,
                        candidate.origin_x,
                        candidate.origin_y,
                        candidate.rotation,
                    )
                    .with_mirrored(candidate.mirror),
                );
            }

            result.boundaries_used = if result.placements.is_empty() { 0 } else { 1 };
            // Area of the pieces that were placed. It used to sum every piece
            // handed in, so a layout reported the utilization it would have
            // had if all of them had fitted.
            let placed_area: f64 = solution
                .assignments
                .iter()
                .map(|(piece_idx, _)| pieces[*piece_idx].area)
                .sum();
            result.utilization = placed_area / (bound_width * bound_height);

            // The model may now leave a piece out, so the ones left out are
            // said rather than silently missing from a layout that looks
            // complete.
            let placed: std::collections::HashSet<usize> = solution
                .assignments
                .iter()
                .map(|(piece_idx, _)| *piece_idx)
                .collect();
            for (idx, piece) in pieces.iter().enumerate() {
                if !placed.contains(&idx) {
                    result.unplaced.push(piece.id.clone());
                }
            }
            result.best_fitness = Some(solution.objective);
            result.strategy = Some("NfpCm".to_string());
            result.iterations = Some(solution.exact_result.iterations);

            if solution.exact_result.is_optimal {
                log::info!("NFP-CM found optimal solution");
            }
        }
        None => {
            log::warn!("NFP-CM solver failed");
            for piece in &pieces {
                result.unplaced.push(piece.id.clone());
            }
        }
    }

    result.computation_time_ms = start.elapsed().as_millis() as u64;
    result
}

/// Build piece info with candidate positions.
fn build_piece_info(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    config: &Config,
    rotation_angles: &[f64],
    grid_step: f64,
    cancelled: &Arc<AtomicBool>,
) -> Vec<PieceInfo> {
    let (b_min, b_max) = boundary.aabb();
    let margin = config.margin;

    let mut pieces = Vec::new();

    for (geom_idx, geom) in geometries.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            return pieces;
        }

        // Compute dimensions at each rotation
        let mut widths = Vec::new();
        let mut heights = Vec::new();
        for &angle in rotation_angles {
            let (g_min, g_max) = geom.aabb_at_rotation(angle);
            widths.push(g_max[0] - g_min[0]);
            heights.push(g_max[1] - g_min[1]);
        }

        // Mirror candidates (`allow_flip` support): mirroring preserves a
        // polygon's AABB width/height (see `nester.rs`'s `mirror_candidates`
        // doc comment for the same fact applied to the BLF strategy), so the
        // same position grid applies to both — only the tag differs. Doubles
        // the candidate (and later conflict-pair) count when enabled, which
        // matters more here than for the metaheuristic strategies: this
        // solver's own doc comment already caps it at small instances
        // (≤15-20 pieces) due to grid discretization + MILP complexity.
        let mirror_candidates: &[bool] = if geom.allow_flip() {
            &[false, true]
        } else {
            &[false]
        };

        for instance in 0..geom.quantity() {
            let mut candidates = Vec::new();

            // Generate candidate positions for each rotation
            for (rot_idx, &angle) in rotation_angles.iter().enumerate() {
                let w = widths[rot_idx];
                let h = heights[rot_idx];

                // Generate grid of positions where piece fits
                let min_x = b_min[0] + margin;
                let max_x = b_max[0] - margin - w;
                let min_y = b_min[1] + margin;
                let max_y = b_max[1] - margin - h;

                if max_x < min_x || max_y < min_y {
                    continue; // Piece doesn't fit at this rotation
                }

                // AABB-min-to-origin offset, once per (rotation, mirror) —
                // mirroring can shift it even though it never changes
                // width/height (see the `mirror_candidates` comment above).
                let origin_offsets: Vec<(bool, [f64; 2])> = mirror_candidates
                    .iter()
                    .map(|&mirror| {
                        let (g_min, _) = geom.aabb_at_rotation_mirrored(angle, mirror);
                        (mirror, g_min)
                    })
                    .collect();

                // Sample positions on grid
                let mut x = min_x;
                while x <= max_x {
                    let mut y = min_y;
                    while y <= max_y {
                        for &(mirror, g_min) in &origin_offsets {
                            candidates.push(CandidatePosition {
                                x,
                                y,
                                origin_x: x - g_min[0],
                                origin_y: y - g_min[1],
                                rotation: angle,
                                rotation_idx: rot_idx,
                                mirror,
                            });
                        }
                        y += grid_step;
                    }
                    x += grid_step;
                }
            }

            // Limit candidates to avoid explosion
            if candidates.len() > 1000 {
                // Sample uniformly
                let step = candidates.len() / 1000;
                candidates = candidates.into_iter().step_by(step).collect();
            }

            if !candidates.is_empty() {
                pieces.push(PieceInfo {
                    geometry_idx: geom_idx,
                    instance_num: instance,
                    id: geom.id().to_string(),
                    area: geom.measure(),
                    widths: widths.clone(),
                    heights: heights.clone(),
                    candidates,
                });
            }
        }
    }

    pieces
}

/// Conflict between two (piece, candidate) pairs.
type Conflict = ((usize, usize), (usize, usize));

/// NFP cache key: (geometry_a, geometry_b, rotation_idx_a, rotation_idx_b,
/// mirror_a, mirror_b) — mirror flags included since a mirrored and an
/// unmirrored NFP for the same (geometries, rotations) pair are different
/// polygons (`allow_flip` support).
type NfpCacheKey = (usize, usize, usize, usize, bool, bool);

/// Compute conflicts between candidates using NFPs.
fn compute_conflicts(
    pieces: &[PieceInfo],
    geometries: &[Geometry2D],
    spacing: f64,
    cancelled: &Arc<AtomicBool>,
    start: Instant,
    time_limit_ms: u64,
) -> Vec<Conflict> {
    let mut conflicts = Vec::new();

    // NFP cache — see `NfpCacheKey` doc comment.
    let mut nfp_cache: HashMap<NfpCacheKey, Option<Nfp>> = HashMap::new();

    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            if cancelled.load(Ordering::Relaxed) {
                return conflicts;
            }

            // Time limit check
            if start.elapsed().as_millis() as u64 > time_limit_ms / 4 {
                log::warn!("Conflict computation taking too long, using simplified model");
                // Use simple AABB overlap check instead. It is coarser than the
                // no-fit polygon -- it refuses placements the real shapes would
                // allow -- but it has to keep `spacing`, or the fallback would
                // permit what the caller asked to be kept apart.
                return compute_aabb_conflicts(pieces, spacing);
            }

            let geom_i = &geometries[pieces[i].geometry_idx];
            let geom_j = &geometries[pieces[j].geometry_idx];

            for (ci, cand_i) in pieces[i].candidates.iter().enumerate() {
                for (cj, cand_j) in pieces[j].candidates.iter().enumerate() {
                    // Check if these two placements conflict
                    let cache_key = (
                        pieces[i].geometry_idx,
                        pieces[j].geometry_idx,
                        cand_i.rotation_idx,
                        cand_j.rotation_idx,
                        cand_i.mirror,
                        cand_j.mirror,
                    );

                    // Grown by `spacing` here, once per cache entry, with the
                    // crate's own offset. Growing commutes with the rigid
                    // motion applied below, so doing it on the cached local
                    // NFP is the same as doing it on the absolute one.
                    let nfp_opt = nfp_cache.entry(cache_key).or_insert_with(|| {
                        compute_nfp_mirrored(
                            geom_i,
                            geom_j,
                            cand_j.rotation - cand_i.rotation,
                            cand_i.mirror,
                            cand_j.mirror,
                        )
                        .ok()
                        .map(|nfp| offset_nfp(&nfp, spacing))
                    });

                    let overlaps = if let Some(nfp) = nfp_opt {
                        // `nfp` is expressed in `geom_i`'s own *local* frame
                        // (it was computed with `geom_i` fixed at rotation
                        // 0, per `compute_nfp_mirrored`'s contract) — valid
                        // only while `cand_i` itself sits at rotation 0 in
                        // the real placement. Whenever `cand_i` has its own
                        // non-zero absolute rotation, testing a raw global
                        // offset against this local-frame NFP silently
                        // mismatches the two candidates' actual geometry.
                        // Every strategy that already places pieces
                        // correctly (`nester.rs`'s NFP-guided placement)
                        // carries the same per-pair NFP into absolute space
                        // the same way: rotate by the stationary piece's own
                        // rotation, then translate to its actual origin —
                        // only then is a candidate's own absolute origin a
                        // valid point to test against it.
                        let absolute_nfp = translate_nfp(
                            &rotate_nfp(nfp, cand_i.rotation),
                            (cand_i.origin_x, cand_i.origin_y),
                        );

                        point_in_nfp(&absolute_nfp, cand_j.origin_x, cand_j.origin_y)
                    } else {
                        // Fallback to AABB check
                        aabb_overlap(
                            cand_i.x,
                            cand_i.y,
                            pieces[i].widths[cand_i.rotation_idx],
                            pieces[i].heights[cand_i.rotation_idx],
                            cand_j.x,
                            cand_j.y,
                            pieces[j].widths[cand_j.rotation_idx],
                            pieces[j].heights[cand_j.rotation_idx],
                            spacing,
                        )
                    };

                    if overlaps {
                        conflicts.push(((i, ci), (j, cj)));
                    }
                }
            }
        }
    }

    conflicts
}

/// Simplified AABB-based conflict computation.
fn compute_aabb_conflicts(pieces: &[PieceInfo], spacing: f64) -> Vec<Conflict> {
    let mut conflicts = Vec::new();

    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            for (ci, cand_i) in pieces[i].candidates.iter().enumerate() {
                for (cj, cand_j) in pieces[j].candidates.iter().enumerate() {
                    if aabb_overlap(
                        cand_i.x,
                        cand_i.y,
                        pieces[i].widths[cand_i.rotation_idx],
                        pieces[i].heights[cand_i.rotation_idx],
                        cand_j.x,
                        cand_j.y,
                        pieces[j].widths[cand_j.rotation_idx],
                        pieces[j].heights[cand_j.rotation_idx],
                        spacing,
                    ) {
                        conflicts.push(((i, ci), (j, cj)));
                    }
                }
            }
        }
    }

    conflicts
}

/// Check AABB overlap with spacing.
fn aabb_overlap(
    x1: f64,
    y1: f64,
    w1: f64,
    h1: f64,
    x2: f64,
    y2: f64,
    w2: f64,
    h2: f64,
    spacing: f64,
) -> bool {
    let overlap_x = x1 < x2 + w2 + spacing && x2 < x1 + w1 + spacing;
    let overlap_y = y1 < y2 + h2 + spacing && y2 < y1 + h1 + spacing;
    overlap_x && overlap_y
}

/// Check if point is inside NFP with spacing buffer.
fn point_in_nfp(nfp: &Nfp, x: f64, y: f64) -> bool {
    nfp.polygons
        .iter()
        .any(|polygon| point_in_polygon(x, y, polygon))
}

/// Point-in-polygon test.
///
/// `spacing` is not a parameter here: it is applied to the no-fit polygon
/// itself, with the crate's own offset, before this runs. It used to be a
/// per-vertex "buffer" applied as `v - buffer.copysign(v)`, which moves every
/// vertex *towards the coordinate origin* -- it shrank the polygon instead of
/// growing it, and did so relative to a point that has nothing to do with the
/// polygon. A shrunken no-fit polygon reports no conflict for placements that
/// really do overlap.
fn point_in_polygon(x: f64, y: f64, polygon: &[(f64, f64)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    // Ray casting algorithm with buffer expansion
    let mut inside = false;
    let n = polygon.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
    }

    inside
}

/// Solve the NFP-CM MILP.
#[cfg(feature = "milp")]
fn solve_nfp_cm_milp(
    pieces: &[PieceInfo],
    conflicts: &[Conflict],
    bound_width: f64,
    origin_x: f64,
    _origin_y: f64,
    _rotation_angles: &[f64],
    config: &ExactConfig,
    cancelled: &Arc<AtomicBool>,
    _start: Instant,
) -> Option<NfpCmSolution> {
    let n = pieces.len();

    // Create problem
    let mut vars = ProblemVariables::new();

    // Binary variables: z[i][c] = 1 if piece i uses candidate position c
    let z: Vec<Vec<Variable>> = pieces
        .iter()
        .enumerate()
        .map(|(i, piece)| {
            piece
                .candidates
                .iter()
                .enumerate()
                .map(|(c, _)| vars.add(variable().binary().name(format!("z_{}_{}", i, c))))
                .collect()
        })
        .collect();

    // Strip length variable
    let strip_length = vars.add(variable().min(0.0).max(bound_width).name("strip_length"));

    // Objective: minimize strip length
    //
    // `config.time_limit_ms` bounds the solver itself (HiGHS), not just the
    // conflict-precomputation heuristic below — without this, a hard
    // instance could run the underlying MIP search indefinitely regardless
    // of what the caller configured.
    // Objective: place as much as fits, and among the layouts that place the
    // same pieces prefer the shortest strip.
    //
    // It used to minimise the strip length alone while requiring every piece
    // to be placed. That is a strip-packing model, and this solver is given a
    // bounded sheet: whenever the pieces did not all fit -- or the search hit
    // its limit without an incumbent -- the model was infeasible and the
    // caller got an empty layout, rather than the pieces that did fit. Every
    // other strategy in this crate places what fits and reports the rest.
    //
    // The tie-break weight is small enough that no layout ever trades a placed
    // piece for a shorter strip: the whole strip term is worth less than the
    // smallest piece.
    let min_area = pieces
        .iter()
        .map(|p| p.area)
        .fold(f64::INFINITY, f64::min)
        .max(f64::MIN_POSITIVE);
    let compactness = 0.5 * min_area / (bound_width + 1.0);
    let placed_area: Expression = pieces
        .iter()
        .enumerate()
        .flat_map(|(i, piece)| z[i].iter().map(move |&v| piece.area * v))
        .sum();
    let mut problem = vars
        .maximise(placed_area - compactness * strip_length)
        .using(default_solver)
        .with_time_limit(config.time_limit_ms as f64 / 1000.0);

    // Constraint: a piece takes at most one position. Not exactly one -- a
    // piece that cannot be placed is reported, not made infeasible.
    for (i, _piece) in pieces.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }

        let sum: Expression = z[i].iter().map(|&v| Expression::from(v)).sum();
        problem = problem.with(constraint!(sum <= 1.0));
    }

    // Constraint: strip length must accommodate all placements
    for (i, piece) in pieces.iter().enumerate() {
        for (c, cand) in piece.candidates.iter().enumerate() {
            // x + width <= strip_length when z[i][c] = 1
            // Linearize: x + width - strip_length <= M * (1 - z[i][c])
            let w = piece.widths[cand.rotation_idx];
            let x_rel = cand.x - origin_x; // Position relative to origin
            let big_m = bound_width * 2.0;

            problem = problem.with(constraint!(
                x_rel + w - strip_length <= big_m * (1.0 - z[i][c])
            ));
        }
    }

    // Constraint: conflicting placements cannot both be selected
    for ((i, ci), (j, cj)) in conflicts {
        // z[i][ci] + z[j][cj] <= 1
        problem = problem.with(constraint!(z[*i][*ci] + z[*j][*cj] <= 1.0));
    }

    // Solve
    log::info!(
        "Solving NFP-CM MILP with {} pieces, {} candidates, {} conflicts",
        n,
        pieces.iter().map(|p| p.candidates.len()).sum::<usize>(),
        conflicts.len()
    );

    match problem.solve() {
        Ok(solution) => {
            let obj_value = solution.value(strip_length);

            // Extract assignments
            let mut assignments = Vec::new();
            for (i, piece) in pieces.iter().enumerate() {
                for (c, _) in piece.candidates.iter().enumerate() {
                    if solution.value(z[i][c]) > 0.5 {
                        assignments.push((i, c));
                        break;
                    }
                }
            }

            let exact_result = ExactResult::optimal(obj_value);

            Some(NfpCmSolution {
                assignments,
                objective: obj_value,
                exact_result,
            })
        }
        Err(e) => {
            log::error!("NFP-CM MILP solver error: {:?}", e);
            None
        }
    }
}

/// Run NFP-CM nesting without the `milp` feature (stub).
#[cfg(not(feature = "milp"))]
pub fn run_nfp_cm_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    _config: &Config,
    _exact_config: &ExactConfig,
    _cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    log::warn!("NFP-CM solver not available (compile with 'milp' feature)");
    let mut result = SolveResult::new();
    for geom in geometries {
        for _ in 0..geom.quantity() {
            result.unplaced.push(geom.id().to_string());
        }
    }
    result.strategy = Some("NfpCm (disabled)".to_string());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_overlap() {
        // Overlapping
        assert!(aabb_overlap(
            0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 10.0, 10.0, 0.0
        ));

        // Not overlapping
        assert!(!aabb_overlap(
            0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 0.0
        ));

        // Overlapping with spacing
        assert!(aabb_overlap(
            0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 1.0
        ));
    }

    #[test]
    fn test_point_in_polygon() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];

        // Inside
        assert!(point_in_polygon(5.0, 5.0, &square));

        // Outside
        assert!(!point_in_polygon(15.0, 5.0, &square));
    }

    /// Two pieces on a sheet that holds them, at the grid the strategy now
    /// derives from the pieces. With the fixed one-unit grid this used to run
    /// on, each piece carried about a thousand candidates and every pair of
    /// them became a constraint, so the search returned an empty layout for
    /// anything past a single piece however long it was given.
    #[test]
    #[cfg(feature = "milp")]
    fn two_pieces_that_fit_land_on_the_grid_the_pieces_imply() {
        let geometries = vec![Geometry2D::rectangle("R", 50.0, 50.0).with_quantity(2)];
        let boundary = Boundary2D::rectangle(200.0, 100.0);
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(60_000)
            .with_rotation_steps(1)
            // A quarter of the pieces' smallest side, as `milp_grid_step` gives.
            .with_grid_step(12.5);

        let result = run_nfp_cm_nesting(
            &geometries,
            &boundary,
            &Config::default(),
            &exact_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert_eq!(
            result.placements.len(),
            2,
            "unplaced: {:?}",
            result.unplaced
        );
    }

    /// Three pieces offered a sheet that holds one. The model used to require
    /// every piece to be placed, so this was infeasible and the caller got an
    /// empty layout; it now places the one that fits and says which did not.
    /// Utilization counts what was placed -- it used to sum every piece handed
    /// in, reporting the number the layout would have had if all had fitted.
    #[test]
    #[cfg(feature = "milp")]
    fn what_does_not_fit_is_reported_rather_than_making_the_model_infeasible() {
        let geometries = vec![Geometry2D::rectangle("R", 50.0, 50.0).with_quantity(3)];
        let boundary = Boundary2D::rectangle(60.0, 60.0);
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(60_000)
            .with_rotation_steps(1)
            .with_grid_step(5.0);

        let result = run_nfp_cm_nesting(
            &geometries,
            &boundary,
            &Config::default(),
            &exact_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert_eq!(
            result.placements.len(),
            1,
            "unplaced: {:?}",
            result.unplaced
        );
        assert!(
            !result.unplaced.is_empty(),
            "the pieces that did not fit are not reported"
        );
        let expected = 2500.0 / (60.0 * 60.0);
        assert!(
            (result.utilization - expected).abs() < 1e-9,
            "utilization {} should be the one placed piece, {expected}",
            result.utilization
        );
    }

    /// `spacing` reaches the conflicts. It used to be applied by moving each
    /// no-fit-polygon vertex *towards the coordinate origin*, which shrinks the
    /// polygon rather than growing it, so placements that overlapped were not
    /// reported as conflicts at all.
    #[test]
    #[cfg(feature = "milp")]
    fn spacing_is_kept_between_placed_pieces() {
        let spacing = 10.0;
        let geometries = vec![Geometry2D::rectangle("R", 20.0, 20.0).with_quantity(2)];
        let boundary = Boundary2D::rectangle(100.0, 40.0);
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(60_000)
            .with_rotation_steps(1)
            .with_grid_step(5.0);

        let result = run_nfp_cm_nesting(
            &geometries,
            &boundary,
            &Config::default().with_spacing(spacing),
            &exact_config,
            Arc::new(AtomicBool::new(false)),
        );

        assert_eq!(
            result.placements.len(),
            2,
            "unplaced: {:?}",
            result.unplaced
        );
        let a = &result.placements[0];
        let b = &result.placements[1];
        // Axis-aligned 20x20 squares: the gap along the axis they are apart on.
        let gap_x = (a.x() - b.x()).abs() - 20.0;
        let gap_y = (a.y() - b.y()).abs() - 20.0;
        let gap = gap_x.max(gap_y);
        assert!(
            gap >= spacing - 1e-6,
            "pieces at ({}, {}) and ({}, {}) are {gap} apart, less than the {spacing} asked for",
            a.x(),
            a.y(),
            b.x(),
            b.y()
        );
    }

    #[test]
    #[cfg(feature = "milp")]
    fn test_nfp_cm_simple() {
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(2)];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(10000)
            .with_rotation_steps(1)
            .with_grid_step(5.0); // Coarse grid for faster test

        let cancelled = Arc::new(AtomicBool::new(false));
        let result = run_nfp_cm_nesting(&geometries, &boundary, &config, &exact_config, cancelled);

        // Should find a solution
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_build_piece_info_mirror_candidates_only_when_allowed() {
        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let rotation_angles = vec![0.0];
        let cancelled = Arc::new(AtomicBool::new(false));

        let plain = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(1)];
        let pieces = build_piece_info(
            &plain,
            &boundary,
            &config,
            &rotation_angles,
            10.0,
            &cancelled,
        );
        assert!(
            pieces[0].candidates.iter().all(|c| !c.mirror),
            "no candidate should be tagged mirrored without allow_flip"
        );

        let flippable = vec![Geometry2D::rectangle("R1", 10.0, 10.0)
            .with_flip(true)
            .with_quantity(1)];
        let pieces = build_piece_info(
            &flippable,
            &boundary,
            &config,
            &rotation_angles,
            10.0,
            &cancelled,
        );
        assert!(
            pieces[0].candidates.iter().any(|c| c.mirror),
            "allow_flip must produce at least one mirrored candidate"
        );
        assert!(
            pieces[0].candidates.iter().any(|c| !c.mirror),
            "allow_flip must still keep the unmirrored candidates too"
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

    /// `allow_flip`/mirroring, NFP-CM (MILP) strategy — the last of the 7
    /// placement strategies. `run_nfp_cm_nesting` calls no `.validate()`, so
    /// this is a genuine end-to-end run (not a lower-level bypass).
    #[test]
    #[cfg(feature = "milp")]
    fn test_nfp_cm_mirror_no_overlap() {
        use crate::nfp::PlacedGeometry;

        let geometries = vec![chiral_l("L").with_flip(true).with_quantity(2)];
        let boundary = Boundary2D::rectangle(65.0, 45.0);
        let config = Config::default().with_spacing(1.0);
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(15000)
            .with_rotation_steps(1)
            .with_grid_step(5.0);

        let cancelled = Arc::new(AtomicBool::new(false));
        let result = run_nfp_cm_nesting(&geometries, &boundary, &config, &exact_config, cancelled);

        assert_eq!(
            result.placements.len(),
            2,
            "both instances should fit in this boundary"
        );

        let polys: Vec<Vec<(f64, f64)>> = result
            .placements
            .iter()
            .map(|p| {
                PlacedGeometry::new(geometries[0].clone(), (p.x(), p.y()), p.angle())
                    .with_mirrored(p.mirrored)
                    .translated_exterior()
            })
            .collect();
        assert!(
            !polygons_overlap(&polys[0], &polys[1]),
            "placements must not overlap regardless of mirror state (mirrored: {}, {})",
            result.placements[0].mirrored,
            result.placements[1].mirrored
        );
    }
}
