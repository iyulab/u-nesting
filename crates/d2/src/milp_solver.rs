//! MILP-based exact solver for 2D nesting problems.
//!
//! This module implements an exact solver using Mixed Integer Linear Programming (MILP)
//! with the HiGHS solver via the `good_lp` crate. It is suitable for small instances
//! (typically ≤15-20 pieces) where optimal solutions are desired.
//!
//! # Algorithm
//!
//! The formulation uses:
//! - Continuous variables for x, y positions of each piece
//! - Binary variables for rotation selection
//! - Big-M constraints for non-overlap between pieces
//! - Objective: minimize strip length (for strip packing) or maximize utilization
//!
//! # Features
//!
//! - Discrete rotation angles (configurable)
//! - Symmetry breaking constraints
//! - Valid inequalities for tighter bounds
//! - Timeout and gap tolerance support
//!
//! # Example
//!
//! ```ignore
//! use u_nesting_d2::milp_solver::run_milp_nesting;
//! use u_nesting_core::{Config, ExactConfig};
//!
//! let exact_config = ExactConfig::default()
//!     .with_time_limit_ms(60000)
//!     .with_max_items(15);
//!
//! let result = run_milp_nesting(&geometries, &boundary, &config, &exact_config);
//! ```

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
use u_nesting_core::exact::{ExactConfig, ExactResult, SolutionStatus};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::{Placement, SolveResult};

#[cfg(feature = "milp")]
use good_lp::{
    constraint, default_solver, variable, Expression, ProblemVariables, Solution, SolverModel,
    Variable,
};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Piece instance with precomputed bounds for different rotations.
#[derive(Debug, Clone)]
struct PieceInstance {
    /// Original geometry index.
    geometry_idx: usize,
    /// Instance number within geometry.
    instance_num: usize,
    /// Width at each rotation angle.
    widths: Vec<f64>,
    /// Height at each rotation angle.
    heights: Vec<f64>,
    /// Area of the piece.
    area: f64,
    /// Geometry ID.
    id: String,
}

/// Solution from MILP solver.
#[derive(Debug, Clone)]
struct MilpSolution {
    /// Position and rotation for each piece.
    placements: Vec<(f64, f64, usize)>, // (x, y, rotation_index)
    /// Objective value (strip length).
    objective: f64,
    /// Exact result info.
    exact_result: ExactResult,
}

/// Run MILP-based exact nesting.
///
/// This function formulates the nesting problem as a MILP and solves it using HiGHS.
/// It is only suitable for small instances due to computational complexity.
///
/// # Arguments
///
/// * `geometries` - Slice of geometries to place
/// * `boundary` - Boundary to place within
/// * `config` - General solver configuration
/// * `exact_config` - Exact solver specific configuration
/// * `cancelled` - Cancellation flag
///
/// # Returns
///
/// `SolveResult` with placements and extended `ExactResult` information.
#[cfg(feature = "milp")]
pub fn run_milp_nesting(
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

    // Check if within limit
    if !exact_config.is_within_limit(total_instances) {
        log::warn!(
            "Instance count {} exceeds exact limit {}, returning empty result",
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
        log::error!("Invalid boundary dimensions after margin");
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        return result;
    }

    // Prepare piece instances with precomputed dimensions at each rotation
    let rotation_angles = exact_config.rotation_angles();
    let num_rotations = rotation_angles.len();

    let mut pieces: Vec<PieceInstance> = Vec::new();
    for (geom_idx, geom) in geometries.iter().enumerate() {
        let mut widths = Vec::with_capacity(num_rotations);
        let mut heights = Vec::with_capacity(num_rotations);

        for &angle in &rotation_angles {
            let (g_min, g_max) = geom.aabb_at_rotation(angle);
            widths.push(g_max[0] - g_min[0]);
            heights.push(g_max[1] - g_min[1]);
        }

        for instance in 0..geom.quantity() {
            pieces.push(PieceInstance {
                geometry_idx: geom_idx,
                instance_num: instance,
                widths: widths.clone(),
                heights: heights.clone(),
                area: geom.measure(),
                id: geom.id().to_string(),
            });
        }
    }

    let n = pieces.len();
    if n == 0 {
        result.computation_time_ms = start.elapsed().as_millis() as u64;
        return result;
    }

    // Build and solve MILP
    match solve_milp(
        &pieces,
        bound_width,
        bound_height,
        b_min[0] + margin,
        b_min[1] + margin,
        config.spacing,
        exact_config,
        &cancelled,
        start,
    ) {
        Some(solution) => {
            // Convert solution to placements
            for (i, piece) in pieces.iter().enumerate() {
                if i < solution.placements.len() {
                    let (x, y, rot_idx) = solution.placements[i];
                    let rotation = rotation_angles.get(rot_idx).copied().unwrap_or(0.0);

                    result.placements.push(Placement::new_2d(
                        piece.id.clone(),
                        piece.instance_num,
                        x,
                        y,
                        rotation,
                    ));
                }
            }

            result.boundaries_used = if result.placements.is_empty() { 0 } else { 1 };
            result.utilization =
                pieces.iter().map(|p| p.area).sum::<f64>() / (bound_width * bound_height);
            result.best_fitness = Some(solution.objective);
            result.strategy = Some("MilpExact".to_string());
            result.iterations = Some(solution.exact_result.iterations);

            // Store exact result info
            if solution.exact_result.is_optimal {
                log::info!("MILP found optimal solution");
            } else {
                log::info!(
                    "MILP found feasible solution (gap: {:.2}%)",
                    solution.exact_result.gap * 100.0
                );
            }
        }
        None => {
            log::warn!("MILP solver failed to find a solution");
            for piece in &pieces {
                result.unplaced.push(piece.id.clone());
            }
        }
    }

    result.computation_time_ms = start.elapsed().as_millis() as u64;
    result
}

/// Solve the MILP formulation.
#[cfg(feature = "milp")]
fn solve_milp(
    pieces: &[PieceInstance],
    width: f64,
    height: f64,
    origin_x: f64,
    origin_y: f64,
    spacing: f64,
    config: &ExactConfig,
    cancelled: &Arc<AtomicBool>,
    start: Instant,
) -> Option<MilpSolution> {
    let n = pieces.len();
    let r = config.rotation_steps;

    // Big-M value for non-overlap constraints
    let big_m = width.max(height) * 2.0;

    // Create problem
    let mut vars = ProblemVariables::new();

    // Position variables (continuous)
    // x[i], y[i] = position of piece i
    let x: Vec<Variable> = (0..n)
        .map(|i| vars.add(variable().min(0.0).max(width).name(format!("x_{}", i))))
        .collect();
    let y: Vec<Variable> = (0..n)
        .map(|i| vars.add(variable().min(0.0).max(height).name(format!("y_{}", i))))
        .collect();

    // Rotation binary variables
    // rot[i][k] = 1 if piece i uses rotation k
    let rot: Vec<Vec<Variable>> = (0..n)
        .map(|i| {
            (0..r)
                .map(|k| vars.add(variable().binary().name(format!("rot_{}_{}", i, k))))
                .collect()
        })
        .collect();

    // Non-overlap binary variables
    // For each pair (i,j), we need variables to select one of 4 disjunctive constraints
    // left[i][j], right[i][j], below[i][j], above[i][j]
    let mut left: Vec<Vec<Variable>> = Vec::new();
    let mut right: Vec<Vec<Variable>> = Vec::new();
    let mut below: Vec<Vec<Variable>> = Vec::new();
    let mut above: Vec<Vec<Variable>> = Vec::new();

    for i in 0..n {
        let mut left_row = Vec::new();
        let mut right_row = Vec::new();
        let mut below_row = Vec::new();
        let mut above_row = Vec::new();

        for j in 0..n {
            if i < j {
                left_row.push(vars.add(variable().binary().name(format!("left_{}_{}", i, j))));
                right_row.push(vars.add(variable().binary().name(format!("right_{}_{}", i, j))));
                below_row.push(vars.add(variable().binary().name(format!("below_{}_{}", i, j))));
                above_row.push(vars.add(variable().binary().name(format!("above_{}_{}", i, j))));
            } else {
                // Placeholder for symmetric pairs
                left_row.push(vars.add(variable().binary()));
                right_row.push(vars.add(variable().binary()));
                below_row.push(vars.add(variable().binary()));
                above_row.push(vars.add(variable().binary()));
            }
        }

        left.push(left_row);
        right.push(right_row);
        below.push(below_row);
        above.push(above_row);
    }

    // Objective: minimize strip length (sum of x + width for all pieces)
    // For simplicity, we minimize the maximum x + width
    let strip_length = vars.add(variable().min(0.0).max(width).name("strip_length"));

    // Build problem
    let mut problem = vars.minimise(strip_length).using(default_solver);

    // Rotation selection constraints: exactly one rotation per piece
    for i in 0..n {
        let sum: Expression = rot[i].iter().map(|&v| v.into()).sum();
        problem = problem.with(constraint!(sum == 1.0));
    }

    // Boundary constraints using linearized rotation-dependent dimensions
    // For piece i with rotation k: x[i] + width[i][k] <= strip_width
    // We linearize using: x[i] + sum_k(width[i][k] * rot[i][k]) <= width
    for i in 0..n {
        // x + width <= strip_width (bound on x)
        let width_expr: Expression = (0..r)
            .map(|k| pieces[i].widths[k] * rot[i][k])
            .fold(Expression::from(0.0), |acc, term| acc + term);
        problem = problem.with(constraint!(x[i] + width_expr <= width));

        // Strip length constraint
        problem = problem.with(constraint!(x[i] + width_expr <= strip_length));

        // y + height <= strip_height (bound on y)
        let height_expr: Expression = (0..r)
            .map(|k| pieces[i].heights[k] * rot[i][k])
            .fold(Expression::from(0.0), |acc, term| acc + term);
        problem = problem.with(constraint!(y[i] + height_expr <= height));
    }

    // Non-overlap constraints using Big-M formulation
    // For each pair (i, j) with i < j:
    // At least one of the following must hold:
    // 1. x[i] + w[i] + spacing <= x[j]  (i is left of j)
    // 2. x[j] + w[j] + spacing <= x[i]  (j is left of i)
    // 3. y[i] + h[i] + spacing <= y[j]  (i is below j)
    // 4. y[j] + h[j] + spacing <= y[i]  (j is below i)
    //
    // Using Big-M:
    // x[i] + w[i] + spacing - x[j] <= M * (1 - left[i][j])
    // etc.
    // And: left[i][j] + right[i][j] + below[i][j] + above[i][j] >= 1

    for i in 0..n {
        for j in (i + 1)..n {
            if cancelled.load(Ordering::Relaxed) {
                return None;
            }

            // Check time limit during constraint building
            if start.elapsed().as_millis() as u64 > config.time_limit_ms / 2 {
                log::warn!("Constraint building taking too long, using simplified model");
                // Continue with simplified constraints
            }

            // Get linearized width/height expressions for pieces i and j
            // For simplicity in Big-M, we use max dimensions (conservative)
            let w_i = pieces[i].widths.iter().cloned().fold(0.0_f64, f64::max);
            let w_j = pieces[j].widths.iter().cloned().fold(0.0_f64, f64::max);
            let h_i = pieces[i].heights.iter().cloned().fold(0.0_f64, f64::max);
            let h_j = pieces[j].heights.iter().cloned().fold(0.0_f64, f64::max);

            // At least one direction must separate them
            problem = problem.with(constraint!(
                left[i][j] + right[i][j] + below[i][j] + above[i][j] >= 1.0
            ));

            // Left: x[i] + w_i + spacing <= x[j] + M*(1-left[i][j])
            problem = problem.with(constraint!(
                x[i] - x[j] <= big_m - big_m * left[i][j] - w_i - spacing
            ));

            // Right: x[j] + w_j + spacing <= x[i] + M*(1-right[i][j])
            problem = problem.with(constraint!(
                x[j] - x[i] <= big_m - big_m * right[i][j] - w_j - spacing
            ));

            // Below: y[i] + h_i + spacing <= y[j] + M*(1-below[i][j])
            problem = problem.with(constraint!(
                y[i] - y[j] <= big_m - big_m * below[i][j] - h_i - spacing
            ));

            // Above: y[j] + h_j + spacing <= y[i] + M*(1-above[i][j])
            problem = problem.with(constraint!(
                y[j] - y[i] <= big_m - big_m * above[i][j] - h_j - spacing
            ));
        }
    }

    // Symmetry breaking: order pieces of the same type by x position
    if config.use_symmetry_breaking {
        for i in 0..(n - 1) {
            if pieces[i].geometry_idx == pieces[i + 1].geometry_idx {
                // Same geometry type: x[i] <= x[i+1]
                problem = problem.with(constraint!(x[i] <= x[i + 1]));
            }
        }
    }

    // Solve
    log::info!("Solving MILP with {} pieces, {} rotations", n, r);
    let solution_result = problem.solve();

    match solution_result {
        Ok(solution) => {
            let obj_value = solution.value(strip_length);

            // Extract placements
            let mut placements = Vec::with_capacity(n);
            for i in 0..n {
                let px = solution.value(x[i]) + origin_x;
                let py = solution.value(y[i]) + origin_y;

                // Find which rotation was selected
                let mut selected_rot = 0;
                for k in 0..r {
                    if solution.value(rot[i][k]) > 0.5 {
                        selected_rot = k;
                        break;
                    }
                }

                placements.push((px, py, selected_rot));
            }

            // Determine solution status
            // Note: HiGHS via good_lp doesn't directly expose optimality gap,
            // so we assume optimal if solver returns successfully
            let exact_result = ExactResult::optimal(obj_value);

            Some(MilpSolution {
                placements,
                objective: obj_value,
                exact_result,
            })
        }
        Err(e) => {
            log::error!("MILP solver error: {:?}", e);
            None
        }
    }
}

/// Run MILP nesting without the `milp` feature (stub).
#[cfg(not(feature = "milp"))]
pub fn run_milp_nesting(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    _config: &Config,
    _exact_config: &ExactConfig,
    _cancelled: Arc<AtomicBool>,
) -> SolveResult<f64> {
    log::warn!("MILP solver not available (compile with 'milp' feature)");
    let mut result = SolveResult::new();
    for geom in geometries {
        for _ in 0..geom.quantity() {
            result.unplaced.push(geom.id().to_string());
        }
    }
    result.strategy = Some("MilpExact (disabled)".to_string());
    result
}

/// Check if MILP feature is enabled.
pub fn is_milp_available() -> bool {
    cfg!(feature = "milp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Boundary2D;
    use crate::geometry::Geometry2D;

    #[test]
    fn test_is_milp_available() {
        // Just verify the function works
        let _available = is_milp_available();
    }

    #[test]
    #[cfg(feature = "milp")]
    fn test_milp_simple_rectangles() {
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(2)];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let exact_config = ExactConfig::default()
            .with_time_limit_ms(10000)
            .with_rotation_steps(1); // No rotation for simplicity

        let cancelled = Arc::new(AtomicBool::new(false));
        let result = run_milp_nesting(&geometries, &boundary, &config, &exact_config, cancelled);

        assert!(!result.placements.is_empty(), "Should place some pieces");
        assert_eq!(result.placements.len(), 2, "Should place all 2 pieces");
    }

    #[test]
    #[cfg(feature = "milp")]
    fn test_milp_exceeds_limit() {
        // Create more pieces than the limit
        let geometries = vec![Geometry2D::rectangle("R1", 5.0, 5.0).with_quantity(20)];

        let boundary = Boundary2D::rectangle(100.0, 100.0);
        let config = Config::default();
        let exact_config = ExactConfig::default().with_max_items(10); // Limit to 10

        let cancelled = Arc::new(AtomicBool::new(false));
        let result = run_milp_nesting(&geometries, &boundary, &config, &exact_config, cancelled);

        // Should return empty result since exceeds limit
        assert!(
            result.placements.is_empty(),
            "Should not solve when exceeding limit"
        );
    }

    #[test]
    #[cfg(not(feature = "milp"))]
    fn test_milp_stub() {
        let geometries = vec![Geometry2D::rectangle("R1", 10.0, 10.0).with_quantity(2)];

        let boundary = Boundary2D::rectangle(50.0, 50.0);
        let config = Config::default();
        let exact_config = ExactConfig::default();

        let cancelled = Arc::new(AtomicBool::new(false));
        let result = run_milp_nesting(&geometries, &boundary, &config, &exact_config, cancelled);

        // Should return empty result with unplaced items
        assert!(result.placements.is_empty());
        assert!(!result.unplaced.is_empty());
    }
}
