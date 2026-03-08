//! # U-Nesting WASM
//!
//! WebAssembly bindings for the U-Nesting spatial optimization engine.
//!
//! All functions use JSON string I/O matching the same schema as the C FFI layer.
//!
//! ## Functions
//!
//! - [`solve_2d`] — Solve a 2D nesting problem
//! - [`solve_3d`] — Solve a 3D bin packing problem
//! - [`optimize_cutting_path`] — Optimize cutting path for placed parts
//! - [`version`] — Get API version
//! - [`available_strategies`] — List available strategies for WASM

use u_nesting_core::api_types::*;
use u_nesting_core::solver::{Config, Solver, Strategy};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};
use wasm_bindgen::prelude::*;

/// Solves a 2D nesting problem from a JSON request string.
///
/// Returns a JSON string with the solve result.
/// On error, returns `{ "success": false, "error": "..." }`.
#[wasm_bindgen]
pub fn solve_2d(request_json: &str) -> String {
    let response = solve_2d_internal(request_json);
    serde_json::to_string(&response).unwrap_or_else(|e| {
        format!(
            r#"{{"success":false,"error":"Serialization error: {}"}}"#,
            e
        )
    })
}

/// Solves a 3D bin packing problem from a JSON request string.
///
/// Returns a JSON string with the solve result.
/// On error, returns `{ "success": false, "error": "..." }`.
#[wasm_bindgen]
pub fn solve_3d(request_json: &str) -> String {
    let response = solve_3d_internal(request_json);
    serde_json::to_string(&response).unwrap_or_else(|e| {
        format!(
            r#"{{"success":false,"error":"Serialization error: {}"}}"#,
            e
        )
    })
}

/// Optimizes cutting path for placed 2D parts.
///
/// Input must include geometries, a previous solve result, and optional cutting config.
/// Returns a JSON string with the cutting path result.
/// On error, returns `{ "success": false, "error": "..." }`.
#[wasm_bindgen]
pub fn optimize_cutting_path(request_json: &str) -> String {
    let response = optimize_cutting_path_internal(request_json);
    serde_json::to_string(&response).unwrap_or_else(|e| {
        format!(
            r#"{{"success":false,"error":"Serialization error: {}"}}"#,
            e
        )
    })
}

/// Returns the API version string.
#[wasm_bindgen]
pub fn version() -> String {
    API_VERSION.to_string()
}

/// Returns a JSON array of available strategy names for WASM builds.
///
/// MILP and HybridExact are not available in WASM (requires native HiGHS solver).
#[wasm_bindgen]
pub fn available_strategies() -> String {
    serde_json::to_string(&serde_json::json!({
        "2d": ["blf", "nfp", "ga", "brkga", "sa", "gdrr", "alns"],
        "3d": ["blf", "ep", "ga", "brkga", "sa"]
    }))
    .expect("static JSON serialization should not fail")
}

// --- Internal implementations ---

/// Strategies that are NOT available in WASM builds.
const WASM_BLOCKED_STRATEGIES: &[&str] = &["milp", "milpexact", "hybrid", "hybridexact"];

fn solve_2d_internal(json_str: &str) -> SolveResponse {
    let request: Request2D = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => return SolveResponse::error(format!("Invalid JSON: {e}")),
    };

    // Check for WASM-blocked strategies
    if let Some(ref config) = request.config {
        if let Some(ref strategy) = config.strategy {
            let s = strategy.to_lowercase();
            if WASM_BLOCKED_STRATEGIES
                .iter()
                .any(|blocked| s == *blocked)
            {
                return SolveResponse::error(format!(
                    "Strategy '{strategy}' is not available in WASM builds. \
                     Use 'blf', 'nfp', 'ga', 'brkga', 'sa', 'gdrr', or 'alns'."
                ));
            }
        }
    }

    // Convert geometries
    let geometries: Vec<Geometry2D> = request
        .geometries
        .into_iter()
        .map(|g| {
            let vertices: Vec<(f64, f64)> = g.polygon.into_iter().map(|p| (p[0], p[1])).collect();

            let mut geom = Geometry2D::new(g.id)
                .with_polygon(vertices)
                .with_quantity(g.quantity)
                .with_flip(g.allow_flip);

            if let Some(rotations) = g.rotations {
                geom = geom.with_rotations_deg(rotations);
            }

            if let Some(holes) = g.holes {
                for hole in holes {
                    let hole_vertices: Vec<(f64, f64)> =
                        hole.into_iter().map(|p| (p[0], p[1])).collect();
                    geom = geom.with_hole(hole_vertices);
                }
            }

            geom
        })
        .collect();

    // Convert boundary
    let boundary = if let (Some(w), Some(h)) = (request.boundary.width, request.boundary.height) {
        Boundary2D::rectangle(w, h)
    } else if let Some(polygon) = request.boundary.polygon {
        let vertices: Vec<(f64, f64)> = polygon.into_iter().map(|p| (p[0], p[1])).collect();
        Boundary2D::new(vertices)
    } else {
        return SolveResponse::error("Invalid boundary: specify width/height or polygon");
    };

    // Build config
    let config = build_config(request.config);

    // Solve
    let nester = Nester2D::new(config);
    match nester.solve(&geometries, &boundary) {
        Ok(result) => SolveResponse {
            version: API_VERSION.to_string(),
            success: true,
            error: None,
            placements: result.placements.into_iter().map(Into::into).collect(),
            boundaries_used: result.boundaries_used,
            utilization: result.utilization,
            unplaced: result.unplaced,
            computation_time_ms: result.computation_time_ms,
        },
        Err(e) => SolveResponse::error(e.to_string()),
    }
}

fn solve_3d_internal(json_str: &str) -> SolveResponse {
    let request: Request3D = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => return SolveResponse::error(format!("Invalid JSON: {e}")),
    };

    // Check for WASM-blocked strategies
    if let Some(ref config) = request.config {
        if let Some(ref strategy) = config.strategy {
            let s = strategy.to_lowercase();
            if WASM_BLOCKED_STRATEGIES
                .iter()
                .any(|blocked| s == *blocked)
            {
                return SolveResponse::error(format!(
                    "Strategy '{strategy}' is not available in WASM builds. \
                     Use 'blf', 'ep', 'ga', 'brkga', or 'sa'."
                ));
            }
        }
    }

    // Convert geometries
    let geometries: Vec<Geometry3D> = request
        .geometries
        .into_iter()
        .map(|g| {
            let mut geom =
                Geometry3D::new(g.id, g.dimensions[0], g.dimensions[1], g.dimensions[2])
                    .with_quantity(g.quantity);

            if let Some(mass) = g.mass {
                geom = geom.with_mass(mass);
            }

            geom
        })
        .collect();

    // Convert boundary
    let mut boundary = Boundary3D::new(
        request.boundary.dimensions[0],
        request.boundary.dimensions[1],
        request.boundary.dimensions[2],
    );

    if let Some(max_mass) = request.boundary.max_mass {
        boundary = boundary.with_max_mass(max_mass);
    }

    boundary = boundary
        .with_gravity(request.boundary.gravity)
        .with_stability(request.boundary.stability);

    // Build config
    let config = build_config(request.config);

    // Solve
    let packer = Packer3D::new(config);
    match packer.solve(&geometries, &boundary) {
        Ok(result) => SolveResponse {
            version: API_VERSION.to_string(),
            success: true,
            error: None,
            placements: result.placements.into_iter().map(Into::into).collect(),
            boundaries_used: result.boundaries_used,
            utilization: result.utilization,
            unplaced: result.unplaced,
            computation_time_ms: result.computation_time_ms,
        },
        Err(e) => SolveResponse::error(e.to_string()),
    }
}

fn optimize_cutting_path_internal(json_str: &str) -> CuttingResponse {
    let request: CuttingRequest = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => return CuttingResponse::error(format!("Invalid JSON: {e}")),
    };

    // Validate the solve result
    if !request.solve_result.success {
        return CuttingResponse::error("Solve result indicates failure");
    }

    // Convert geometries
    let geometries: Vec<Geometry2D> = request
        .geometries
        .iter()
        .map(|g| {
            let vertices: Vec<(f64, f64)> = g.polygon.iter().map(|p| (p[0], p[1])).collect();

            let mut geom = Geometry2D::new(&g.id)
                .with_polygon(vertices)
                .with_quantity(g.quantity);

            if let Some(ref holes) = g.holes {
                for hole in holes {
                    let hole_vertices: Vec<(f64, f64)> =
                        hole.iter().map(|p| (p[0], p[1])).collect();
                    geom = geom.with_hole(hole_vertices);
                }
            }

            geom
        })
        .collect();

    // Reconstruct SolveResult from the response
    let mut solve_result = u_nesting_core::SolveResult::<f64>::new();
    for p in &request.solve_result.placements {
        solve_result
            .placements
            .push(u_nesting_core::Placement {
                geometry_id: p.geometry_id.clone(),
                instance: p.instance,
                position: p.position.clone(),
                rotation: p.rotation.clone(),
                boundary_index: p.boundary_index,
                mirrored: false,
                rotation_index: None,
            });
    }
    solve_result.boundaries_used = request.solve_result.boundaries_used;
    solve_result.utilization = request.solve_result.utilization;

    // Build cutting config
    let cutting_config = build_cutting_config(request.cutting_config);

    // Run cutting path optimization
    let result =
        u_nesting_cutting::optimize_cutting_path(&solve_result, &geometries, &cutting_config);

    // Convert result to response
    let sequence: Vec<CutStepResponse> = result
        .sequence
        .iter()
        .map(|step| CutStepResponse {
            contour_id: step.contour_id,
            geometry_id: step.geometry_id.clone(),
            instance: step.instance,
            contour_type: match step.contour_type {
                u_nesting_cutting::ContourType::Exterior => "exterior".to_string(),
                u_nesting_cutting::ContourType::Interior => "interior".to_string(),
            },
            pierce_point: [step.pierce_point.0, step.pierce_point.1],
            cut_direction: match step.cut_direction {
                u_nesting_cutting::CutDirection::Ccw => "ccw".to_string(),
                u_nesting_cutting::CutDirection::Cw => "cw".to_string(),
            },
            rapid_from: step.rapid_from.map(|p| [p.0, p.1]),
            rapid_distance: step.rapid_distance,
            cut_distance: step.cut_distance,
        })
        .collect();

    CuttingResponse {
        version: API_VERSION.to_string(),
        success: true,
        error: None,
        sequence,
        total_cut_distance: result.total_cut_distance,
        total_rapid_distance: result.total_rapid_distance,
        total_pierces: result.total_pierces,
        estimated_time_seconds: result.estimated_time_seconds,
        efficiency: result.efficiency(),
        computation_time_ms: result.computation_time_ms,
    }
}

fn build_config(request: Option<ConfigRequest>) -> Config {
    let mut config = Config::default();

    if let Some(req) = request {
        if let Some(spacing) = req.spacing {
            config.spacing = spacing;
        }
        if let Some(margin) = req.margin {
            config.margin = margin;
        }
        if let Some(time_limit) = req.time_limit_ms {
            config.time_limit_ms = time_limit;
        }
        if let Some(target) = req.target_utilization {
            config.target_utilization = Some(target);
        }
        if let Some(pop) = req.population_size {
            config.population_size = pop;
        }
        if let Some(gens) = req.max_generations {
            config.max_generations = gens;
        }
        if let Some(crossover) = req.crossover_rate {
            config.crossover_rate = crossover;
        }
        if let Some(mutation) = req.mutation_rate {
            config.mutation_rate = mutation;
        }
        if let Some(strategy) = req.strategy {
            config.strategy = parse_strategy(&strategy);
        }
    }

    config
}

fn parse_strategy(s: &str) -> Strategy {
    match s.to_lowercase().as_str() {
        "blf" | "bottomleftfill" => Strategy::BottomLeftFill,
        "nfp" | "nfpguided" => Strategy::NfpGuided,
        "ga" | "genetic" | "geneticalgorithm" => Strategy::GeneticAlgorithm,
        "brkga" => Strategy::Brkga,
        "sa" | "simulatedannealing" => Strategy::SimulatedAnnealing,
        "ep" | "extremepoint" => Strategy::ExtremePoint,
        "gdrr" => Strategy::Gdrr,
        "alns" => Strategy::Alns,
        _ => Strategy::BottomLeftFill,
    }
}

fn build_cutting_config(
    request: Option<CuttingConfigRequest>,
) -> u_nesting_cutting::CuttingConfig {
    let mut config = u_nesting_cutting::CuttingConfig::default();

    if let Some(req) = request {
        if let Some(kerf) = req.kerf_width {
            config.kerf_width = kerf;
        }
        if let Some(weight) = req.pierce_weight {
            config.pierce_weight = weight;
        }
        if let Some(iters) = req.max_2opt_iterations {
            config.max_2opt_iterations = iters;
        }
        if let Some(speed) = req.rapid_speed {
            config.rapid_speed = speed;
        }
        if let Some(speed) = req.cut_speed {
            config.cut_speed = speed;
        }
        if let Some(ref dir) = req.exterior_direction {
            config.exterior_direction = parse_direction(dir);
        }
        if let Some(ref dir) = req.interior_direction {
            config.interior_direction = parse_direction(dir);
        }
        if let Some(home) = req.home_position {
            config.home_position = (home[0], home[1]);
        }
        if let Some(candidates) = req.pierce_candidates {
            config.pierce_candidates = candidates.max(1);
        }
        if let Some(tol) = req.tolerance {
            config.tolerance = tol;
        }
    }

    config
}

fn parse_direction(s: &str) -> u_nesting_cutting::config::CutDirectionPreference {
    match s.to_lowercase().as_str() {
        "ccw" => u_nesting_cutting::config::CutDirectionPreference::Ccw,
        "cw" => u_nesting_cutting::config::CutDirectionPreference::Cw,
        _ => u_nesting_cutting::config::CutDirectionPreference::Auto,
    }
}
