//! Cutting path optimization for 2D nesting results.
//!
//! Given a nesting solve result (placed parts with positions and rotations),
//! this module computes an optimized cutting path that minimizes:
//! - Non-cutting (rapid) travel distance
//! - Number of pierce points
//! - Heat accumulation (optional)
//!
//! # Algorithm
//!
//! 1. **Contour extraction**: Extract cut contours from placed geometries
//! 2. **Hierarchy construction**: Build a DAG of precedence constraints
//!    (holes before exteriors, inner parts before outer parts)
//! 3. **Pierce point selection**: Choose optimal entry points on each contour
//! 4. **Sequence optimization**: TSP-based ordering with precedence constraints
//!    (Nearest Neighbor + constrained 2-opt)
//! 5. **Path assembly**: Combine into final cutting path with rapid moves
//!
//! # References
//!
//! - Dewil et al. (2016), "A review of cutting path algorithms for laser cutters"
//! - Hu, Lin & Fu (2023), "Optimizing Cutting Sequences for Common-Edge Nested Parts"

pub mod bridge;
pub mod common_edge;
pub mod config;
pub mod contour;
pub mod cost;
pub mod gtsp;
pub mod hierarchy;
pub mod kerf;
pub mod leadin;
pub mod path;
pub mod pierce;
pub mod result;
pub mod sequence;
pub mod thermal;

pub use config::CuttingConfig;
pub use contour::ContourType;
pub use path::optimize_cutting_path;
pub use result::{CutDirection, CutStep, CuttingPathResult, SkipReason, SkippedContour};

/// Optimizes the cutting path of a `solve_2d` response -- the one pipeline every
/// binding calls once it has read its request.
///
/// `solve_result` is a successful response (one that is itself a refusal has no
/// layout to cut and is refused first); `geometries` reads the geometries of the
/// problem it answers, and is called after that check so that a request with
/// both faults reports the first; `cutting_config` is the optional `cutting_config`
/// of the request. Every refusal comes back as the response.
#[cfg(feature = "serde")]
pub fn optimize_solved<G, F>(
    solve_result: &u_nesting_core::api_types::SolveResponse,
    geometries: F,
    cutting_config: Option<&u_nesting_core::api_types::CuttingConfigRequest>,
) -> u_nesting_core::api_types::CuttingResponse
where
    G: u_nesting_core::geometry::Geometry2DExt<Scalar = f64>
        + u_nesting_core::geometry::Geometry<Scalar = f64>,
    F: FnOnce() -> u_nesting_core::Result<Vec<G>>,
{
    use u_nesting_core::api_types::CuttingResponse;
    use u_nesting_core::{Error, Placement, SolveResult};

    if !solve_result.success {
        return CuttingResponse::refused(&Error::invalid_option(
            "solve_result",
            "the solve result is a refusal (success: false); give a successful one to cut"
                .to_string(),
        ));
    }
    let geometries = match geometries() {
        Ok(g) => g,
        Err(e) => return CuttingResponse::refused(&e),
    };

    // The layout, read back from the response.
    let mut solved = SolveResult::<f64>::new();
    for p in &solve_result.placements {
        solved.placements.push(Placement {
            geometry_id: p.geometry_id.clone(),
            instance: p.instance,
            position: vec![p.x, p.y],
            rotation: vec![p.rotation.to_radians()],
            boundary_index: p.sheet_index,
            mirrored: p.flipped,
            rotation_index: None,
        });
    }
    solved.boundaries_used = solve_result.sheets_used;
    solved.utilization = solve_result.utilization;

    let config = match CuttingConfig::from_request(cutting_config) {
        Ok(c) => c,
        Err(e) => return CuttingResponse::refused(&e),
    };
    match optimize_cutting_path(&solved, &geometries, &config) {
        Ok(result) => result.to_response(),
        Err(e) => CuttingResponse::refused(&e),
    }
}
