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
use u_nesting_core::geometry::Boundary;
use u_nesting_core::solver::{Config, Solver};
use u_nesting_core::Error;
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting_d3::{Boundary3D, Geometry3D, OrientationConstraint, Packer3D};
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
            r#"{{"success":false,"error":"Serialization error: {}","code":"internal"}}"#,
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
            r#"{{"success":false,"error":"Serialization error: {}","code":"internal"}}"#,
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
            r#"{{"success":false,"error":"Serialization error: {}","code":"internal"}}"#,
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

/// Assembles a success `SolveResponse` from a solved 2D result, deriving every
/// accounting/metric field through the shared `From` impl (single source of
/// truth) and then filling in the placements it leaves empty.
fn build_solve_response(result: u_nesting_core::SolveResult<f64>) -> SolveResponse {
    let placements = result
        .placements
        .iter()
        .cloned()
        .map(PlacementResponse::from)
        .collect();
    let mut resp: SolveResponse = result.into();
    resp.placements = placements;
    resp
}

fn solve_2d_internal(json_str: &str) -> SolveResponse {
    let request: Request2D = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            return SolveResponse::refused(&Error::malformed(None, format!("Invalid JSON: {e}")))
        }
    };

    // Check for WASM-blocked strategies
    if let Some(ref config) = request.config {
        if let Some(ref strategy) = config.strategy {
            let s = strategy.to_lowercase();
            if WASM_BLOCKED_STRATEGIES.iter().any(|blocked| s == *blocked) {
                return SolveResponse::refused(&Error::invalid_option(
                    "strategy",
                    format!(
                        "Strategy '{strategy}' is not available in WASM builds. \
                     Use 'blf', 'nfp', 'ga', 'brkga', 'sa', 'gdrr', or 'alns'."
                    ),
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
    // Exactly one shape: width and height together, or a polygon. Given
    // both, or half of the rectangle, part of the request used to be dropped.
    let boundary = match (
        request.boundary.width,
        request.boundary.height,
        request.boundary.polygon,
    ) {
        (Some(w), Some(h), None) => Boundary2D::rectangle(w, h),
        (None, None, Some(polygon)) => {
            let vertices: Vec<(f64, f64)> = polygon.into_iter().map(|p| (p[0], p[1])).collect();
            Boundary2D::new(vertices)
        }
        _ => {
            return SolveResponse::refused(&Error::invalid_boundary(
                Some("boundary"),
                "give either width and height, or polygon -- not both, not one half".to_string(),
            ))
        }
    };

    // Read the multi-sheet flag before `build_config` consumes the config.
    let multi_sheet = request
        .config
        .as_ref()
        .and_then(|c| c.multi_sheet)
        .unwrap_or(false);

    // Build config
    let config = match request
        .config
        .as_ref()
        .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config)
    {
        Ok(c) => c,
        Err(e) => return SolveResponse::refused(&e),
    };

    // Solve — `multi_sheet` distributes overflow across additional sheets.
    let nester = Nester2D::new(config);
    let solved = if multi_sheet {
        nester.solve_multi_strip(&geometries, &boundary)
    } else {
        nester.solve(&geometries, &boundary)
    };
    match solved {
        Ok(mut result) => {
            if multi_sheet {
                // `solve_multi_strip` emits global strip coordinates; localize to the
                // per-sheet frame so each placement's x is relative to its own sheet.
                let (b_min, b_max) = boundary.aabb();
                result.to_boundary_local(b_max[0] - b_min[0]);
            }
            build_solve_response(result)
        }
        Err(e) => SolveResponse::refused(&e),
    }
}

fn solve_3d_internal(json_str: &str) -> Pack3DResponse {
    let request: Request3D = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            return Pack3DResponse::refused(&Error::malformed(None, format!("Invalid JSON: {e}")))
        }
    };

    // Check for WASM-blocked strategies
    if let Some(ref config) = request.config {
        if let Some(ref strategy) = config.strategy {
            let s = strategy.to_lowercase();
            if WASM_BLOCKED_STRATEGIES.iter().any(|blocked| s == *blocked) {
                return Pack3DResponse::refused(&Error::invalid_option(
                    "strategy",
                    format!(
                        "Strategy '{strategy}' is not available in WASM builds. \
                     Use 'blf', 'ep', 'ga', 'brkga', or 'sa'."
                    ),
                ));
            }
        }
    }

    // Convert geometries
    let geometries: u_nesting_core::Result<Vec<Geometry3D>> = request
        .geometries
        .into_iter()
        .enumerate()
        .map(|(i, g)| {
            let mut geom = Geometry3D::new(g.id, g.dimensions[0], g.dimensions[1], g.dimensions[2])
                .with_quantity(g.quantity);

            if let Some(mass) = g.mass {
                geom = geom.with_mass(mass);
            }
            if let Some(name) = g.orientation {
                let constraint =
                    OrientationConstraint::parse(&name).ok_or_else(|| Error::UnknownOption {
                        parameter: format!("geometries[{i}].orientation"),
                        got: name.clone(),
                        expected: vec!["any".into(), "upright".into(), "fixed".into()],
                    })?;
                geom = geom.with_orientation(constraint);
            }

            Ok(geom)
        })
        .collect();
    let geometries = match geometries {
        Ok(g) => g,
        Err(e) => return Pack3DResponse::refused(&e),
    };

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
    let config = match request
        .config
        .as_ref()
        .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config)
    {
        Ok(c) => c,
        Err(e) => return Pack3DResponse::refused(&e),
    };

    // Solve
    let packer = Packer3D::new(config);
    match packer.solve(&geometries, &boundary) {
        Ok(result) => u_nesting_d3::build_pack3d_response(&result, &geometries),
        Err(e) => Pack3DResponse::refused(&e),
    }
}

fn optimize_cutting_path_internal(json_str: &str) -> CuttingResponse {
    let request: CuttingRequest = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            return CuttingResponse::refused(&Error::malformed(None, format!("Invalid JSON: {e}")))
        }
    };

    // Validate the solve result
    if !request.solve_result.success {
        return CuttingResponse::refused(&Error::invalid_option(
            "solve_result",
            "the solve result is a refusal (success: false); give a successful one to cut"
                .to_string(),
        ));
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
        solve_result.placements.push(u_nesting_core::Placement {
            geometry_id: p.geometry_id.clone(),
            instance: p.instance,
            position: vec![p.x, p.y],
            rotation: vec![p.rotation.to_radians()],
            boundary_index: p.sheet_index,
            mirrored: p.flipped,
            rotation_index: None,
        });
    }
    solve_result.boundaries_used = request.solve_result.sheets_used;
    solve_result.utilization = request.solve_result.utilization;

    // Build cutting config
    let cutting_config =
        match u_nesting_cutting::CuttingConfig::from_request(request.cutting_config.as_ref()) {
            Ok(c) => c,
            Err(e) => return CuttingResponse::refused(&e),
        };

    // Run cutting path optimization
    let result =
        match u_nesting_cutting::optimize_cutting_path(&solve_result, &geometries, &cutting_config)
        {
            Ok(r) => r,
            Err(e) => return CuttingResponse::refused(&e),
        };

    result.to_response()
}
