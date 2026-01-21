//! C FFI API functions.

use crate::types::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use u_nesting_core::solver::{Config, Solver, Strategy};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};

/// Error codes.
pub const UNESTING_OK: i32 = 0;
pub const UNESTING_ERR_NULL_PTR: i32 = -1;
pub const UNESTING_ERR_INVALID_JSON: i32 = -2;
pub const UNESTING_ERR_SOLVE_FAILED: i32 = -3;
pub const UNESTING_ERR_UNKNOWN: i32 = -99;

/// Solves a 2D nesting problem from JSON input.
///
/// # Safety
/// - `request_json` must be a valid null-terminated UTF-8 string
/// - `result_ptr` must be a valid pointer to a `*mut c_char`
/// - The caller must free the result string using `unesting_free_string`
#[no_mangle]
pub unsafe extern "C" fn unesting_solve_2d(
    request_json: *const c_char,
    result_ptr: *mut *mut c_char,
) -> i32 {
    if request_json.is_null() || result_ptr.is_null() {
        return UNESTING_ERR_NULL_PTR;
    }

    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => return UNESTING_ERR_INVALID_JSON,
    };

    let response = solve_2d_internal(json_str);
    let response_json = match serde_json::to_string(&response) {
        Ok(s) => s,
        Err(_) => return UNESTING_ERR_UNKNOWN,
    };

    match CString::new(response_json) {
        Ok(cstr) => {
            *result_ptr = cstr.into_raw();
            if response.success {
                UNESTING_OK
            } else {
                UNESTING_ERR_SOLVE_FAILED
            }
        }
        Err(_) => UNESTING_ERR_UNKNOWN,
    }
}

/// Solves a 3D bin packing problem from JSON input.
///
/// # Safety
/// - `request_json` must be a valid null-terminated UTF-8 string
/// - `result_ptr` must be a valid pointer to a `*mut c_char`
/// - The caller must free the result string using `unesting_free_string`
#[no_mangle]
pub unsafe extern "C" fn unesting_solve_3d(
    request_json: *const c_char,
    result_ptr: *mut *mut c_char,
) -> i32 {
    if request_json.is_null() || result_ptr.is_null() {
        return UNESTING_ERR_NULL_PTR;
    }

    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => return UNESTING_ERR_INVALID_JSON,
    };

    let response = solve_3d_internal(json_str);
    let response_json = match serde_json::to_string(&response) {
        Ok(s) => s,
        Err(_) => return UNESTING_ERR_UNKNOWN,
    };

    match CString::new(response_json) {
        Ok(cstr) => {
            *result_ptr = cstr.into_raw();
            if response.success {
                UNESTING_OK
            } else {
                UNESTING_ERR_SOLVE_FAILED
            }
        }
        Err(_) => UNESTING_ERR_UNKNOWN,
    }
}

/// Auto-detects mode and solves from JSON input.
///
/// # Safety
/// - `request_json` must be a valid null-terminated UTF-8 string
/// - `result_ptr` must be a valid pointer to a `*mut c_char`
/// - The caller must free the result string using `unesting_free_string`
#[no_mangle]
pub unsafe extern "C" fn unesting_solve(
    request_json: *const c_char,
    result_ptr: *mut *mut c_char,
) -> i32 {
    if request_json.is_null() || result_ptr.is_null() {
        return UNESTING_ERR_NULL_PTR;
    }

    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => return UNESTING_ERR_INVALID_JSON,
    };

    // Try to detect mode from JSON
    let value: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return UNESTING_ERR_INVALID_JSON,
    };

    let mode = value.get("mode").and_then(|m| m.as_str()).unwrap_or("2d");

    let response = match mode {
        "3d" => solve_3d_internal(json_str),
        _ => solve_2d_internal(json_str),
    };

    let response_json = match serde_json::to_string(&response) {
        Ok(s) => s,
        Err(_) => return UNESTING_ERR_UNKNOWN,
    };

    match CString::new(response_json) {
        Ok(cstr) => {
            *result_ptr = cstr.into_raw();
            if response.success {
                UNESTING_OK
            } else {
                UNESTING_ERR_SOLVE_FAILED
            }
        }
        Err(_) => UNESTING_ERR_UNKNOWN,
    }
}

/// Frees a string allocated by U-Nesting.
///
/// # Safety
/// - `ptr` must have been allocated by a U-Nesting function
/// - `ptr` must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn unesting_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Returns the API version from Cargo.toml.
///
/// # Safety
/// - The returned string is statically allocated and must not be freed
#[no_mangle]
pub extern "C" fn unesting_version() -> *const c_char {
    // Use version from Cargo.toml at compile time
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const c_char
}

// Internal implementation functions

fn solve_2d_internal(json_str: &str) -> SolveResponse {
    let request: Request2D = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            return SolveResponse {
                version: API_VERSION.to_string(),
                success: false,
                error: Some(format!("Invalid JSON: {}", e)),
                placements: Vec::new(),
                boundaries_used: 0,
                utilization: 0.0,
                unplaced: Vec::new(),
                computation_time_ms: 0,
            };
        }
    };

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
        return SolveResponse {
            version: API_VERSION.to_string(),
            success: false,
            error: Some("Invalid boundary: specify width/height or polygon".into()),
            placements: Vec::new(),
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: Vec::new(),
            computation_time_ms: 0,
        };
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
        Err(e) => SolveResponse {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(e.to_string()),
            placements: Vec::new(),
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: Vec::new(),
            computation_time_ms: 0,
        },
    }
}

fn solve_3d_internal(json_str: &str) -> SolveResponse {
    let request: Request3D = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            return SolveResponse {
                version: API_VERSION.to_string(),
                success: false,
                error: Some(format!("Invalid JSON: {}", e)),
                placements: Vec::new(),
                boundaries_used: 0,
                utilization: 0.0,
                unplaced: Vec::new(),
                computation_time_ms: 0,
            };
        }
    };

    // Convert geometries
    let geometries: Vec<Geometry3D> = request
        .geometries
        .into_iter()
        .map(|g| {
            let mut geom = Geometry3D::new(g.id, g.dimensions[0], g.dimensions[1], g.dimensions[2])
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
        Err(e) => SolveResponse {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(e.to_string()),
            placements: Vec::new(),
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: Vec::new(),
            computation_time_ms: 0,
        },
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
            config.strategy = match strategy.to_lowercase().as_str() {
                "blf" | "bottomleftfill" => Strategy::BottomLeftFill,
                "nfp" | "nfpguided" => Strategy::NfpGuided,
                "ga" | "genetic" | "geneticalgorithm" => Strategy::GeneticAlgorithm,
                "sa" | "simulatedannealing" => Strategy::SimulatedAnnealing,
                "ep" | "extremepoint" => Strategy::ExtremePoint,
                _ => Strategy::BottomLeftFill,
            };
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version_ptr = unesting_version();
        unsafe {
            let version = CStr::from_ptr(version_ptr).to_str().unwrap();
            // Version should match Cargo.toml (env!("CARGO_PKG_VERSION"))
            assert_eq!(version, env!("CARGO_PKG_VERSION"));
        }
    }

    #[test]
    fn test_solve_2d_basic() {
        let request = r#"{
            "geometries": [
                {"id": "rect1", "polygon": [[0,0], [10,0], [10,5], [0,5]], "quantity": 2}
            ],
            "boundary": {"width": 50, "height": 50}
        }"#;

        let request_cstr = CString::new(request).unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve_2d(request_cstr.as_ptr(), &mut result_ptr);
            assert_eq!(code, UNESTING_OK);
            assert!(!result_ptr.is_null());

            let result_str = CStr::from_ptr(result_ptr).to_str().unwrap();
            let response: SolveResponse = serde_json::from_str(result_str).unwrap();

            assert!(response.success);
            assert_eq!(response.placements.len(), 2);
            assert!(response.utilization > 0.0);

            unesting_free_string(result_ptr);
        }
    }

    #[test]
    fn test_solve_3d_basic() {
        let request = r#"{
            "geometries": [
                {"id": "box1", "dimensions": [10, 10, 10], "quantity": 2}
            ],
            "boundary": {"dimensions": [50, 50, 50]}
        }"#;

        let request_cstr = CString::new(request).unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve_3d(request_cstr.as_ptr(), &mut result_ptr);
            assert_eq!(code, UNESTING_OK);
            assert!(!result_ptr.is_null());

            let result_str = CStr::from_ptr(result_ptr).to_str().unwrap();
            let response: SolveResponse = serde_json::from_str(result_str).unwrap();

            assert!(response.success);
            assert_eq!(response.placements.len(), 2);

            unesting_free_string(result_ptr);
        }
    }

    #[test]
    fn test_solve_auto_detect_2d() {
        let request = r#"{
            "mode": "2d",
            "geometries": [
                {"id": "rect1", "polygon": [[0,0], [10,0], [10,5], [0,5]], "quantity": 1}
            ],
            "boundary": {"width": 50, "height": 50}
        }"#;

        let request_cstr = CString::new(request).unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve(request_cstr.as_ptr(), &mut result_ptr);
            assert_eq!(code, UNESTING_OK);
            assert!(!result_ptr.is_null());

            let result_str = CStr::from_ptr(result_ptr).to_str().unwrap();
            let response: SolveResponse = serde_json::from_str(result_str).unwrap();

            assert!(response.success);
            assert_eq!(response.placements.len(), 1);

            unesting_free_string(result_ptr);
        }
    }

    #[test]
    fn test_solve_auto_detect_3d() {
        let request = r#"{
            "mode": "3d",
            "geometries": [
                {"id": "box1", "dimensions": [10, 10, 10], "quantity": 1}
            ],
            "boundary": {"dimensions": [50, 50, 50]}
        }"#;

        let request_cstr = CString::new(request).unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve(request_cstr.as_ptr(), &mut result_ptr);
            assert_eq!(code, UNESTING_OK);
            assert!(!result_ptr.is_null());

            let result_str = CStr::from_ptr(result_ptr).to_str().unwrap();
            let response: SolveResponse = serde_json::from_str(result_str).unwrap();

            assert!(response.success);
            assert_eq!(response.placements.len(), 1);

            unesting_free_string(result_ptr);
        }
    }

    #[test]
    fn test_null_pointer_error() {
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            // Null request
            let code = unesting_solve_2d(std::ptr::null(), &mut result_ptr);
            assert_eq!(code, UNESTING_ERR_NULL_PTR);

            // Null result pointer
            let request = CString::new("{}").unwrap();
            let code = unesting_solve_2d(request.as_ptr(), std::ptr::null_mut());
            assert_eq!(code, UNESTING_ERR_NULL_PTR);
        }
    }

    #[test]
    fn test_invalid_json_error() {
        let invalid_json = CString::new("not valid json {{{").unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve_2d(invalid_json.as_ptr(), &mut result_ptr);
            // Should return UNESTING_ERR_SOLVE_FAILED with error message in response
            // or UNESTING_ERR_INVALID_JSON if parsing fails at JSON level
            assert!(code == UNESTING_ERR_SOLVE_FAILED || code == UNESTING_ERR_INVALID_JSON);
        }
    }

    #[test]
    fn test_solve_with_config() {
        let request = r#"{
            "geometries": [
                {"id": "rect1", "polygon": [[0,0], [10,0], [10,5], [0,5]], "quantity": 4}
            ],
            "boundary": {"width": 50, "height": 50},
            "config": {
                "strategy": "blf",
                "spacing": 1.0,
                "margin": 2.0
            }
        }"#;

        let request_cstr = CString::new(request).unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve_2d(request_cstr.as_ptr(), &mut result_ptr);
            assert_eq!(code, UNESTING_OK);

            let result_str = CStr::from_ptr(result_ptr).to_str().unwrap();
            let response: SolveResponse = serde_json::from_str(result_str).unwrap();

            assert!(response.success);
            assert_eq!(response.placements.len(), 4);

            unesting_free_string(result_ptr);
        }
    }

    #[test]
    fn test_solve_2d_with_rotations() {
        let request = r#"{
            "geometries": [
                {"id": "rect1", "polygon": [[0,0], [20,0], [20,5], [0,5]], "quantity": 2, "rotations": [0, 90]}
            ],
            "boundary": {"width": 30, "height": 50}
        }"#;

        let request_cstr = CString::new(request).unwrap();
        let mut result_ptr: *mut c_char = std::ptr::null_mut();

        unsafe {
            let code = unesting_solve_2d(request_cstr.as_ptr(), &mut result_ptr);
            assert_eq!(code, UNESTING_OK);

            let result_str = CStr::from_ptr(result_ptr).to_str().unwrap();
            let response: SolveResponse = serde_json::from_str(result_str).unwrap();

            assert!(response.success);
            // With rotation allowed, both should fit
            assert_eq!(response.placements.len(), 2);

            unesting_free_string(result_ptr);
        }
    }

    #[test]
    fn test_free_null_string() {
        // Should not crash when freeing null
        unsafe {
            unesting_free_string(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_internal_solve_2d() {
        let json = r#"{
            "geometries": [
                {"id": "rect1", "polygon": [[0,0], [10,0], [10,5], [0,5]], "quantity": 1}
            ],
            "boundary": {"width": 50, "height": 50}
        }"#;

        let response = solve_2d_internal(json);
        assert!(response.success);
        assert_eq!(response.placements.len(), 1);
    }

    #[test]
    fn test_internal_solve_3d() {
        let json = r#"{
            "geometries": [
                {"id": "box1", "dimensions": [10, 10, 10], "quantity": 1}
            ],
            "boundary": {"dimensions": [50, 50, 50]}
        }"#;

        let response = solve_3d_internal(json);
        assert!(response.success);
        assert_eq!(response.placements.len(), 1);
    }

    #[test]
    fn test_build_config_default() {
        let config = build_config(None);
        assert_eq!(config.spacing, 0.0);
        assert_eq!(config.margin, 0.0);
    }

    #[test]
    fn test_build_config_with_values() {
        let request = Some(ConfigRequest {
            strategy: Some("nfp".to_string()),
            spacing: Some(2.5),
            margin: Some(1.0),
            time_limit_ms: Some(5000),
            target_utilization: Some(0.8),
            population_size: Some(200),
            max_generations: Some(100),
            crossover_rate: Some(0.9),
            mutation_rate: Some(0.1),
        });

        let config = build_config(request);
        assert_eq!(config.spacing, 2.5);
        assert_eq!(config.margin, 1.0);
        assert_eq!(config.time_limit_ms, 5000);
        assert_eq!(config.target_utilization, Some(0.8));
        assert_eq!(config.population_size, 200);
        assert_eq!(config.max_generations, 100);
        assert_eq!(config.crossover_rate, 0.9);
        assert_eq!(config.mutation_rate, 0.1);
        assert!(matches!(config.strategy, Strategy::NfpGuided));
    }

    #[test]
    fn test_strategy_parsing() {
        let strategies = vec![
            ("blf", Strategy::BottomLeftFill),
            ("bottomleftfill", Strategy::BottomLeftFill),
            ("nfp", Strategy::NfpGuided),
            ("nfpguided", Strategy::NfpGuided),
            ("ga", Strategy::GeneticAlgorithm),
            ("genetic", Strategy::GeneticAlgorithm),
            ("geneticalgorithm", Strategy::GeneticAlgorithm),
            ("sa", Strategy::SimulatedAnnealing),
            ("simulatedannealing", Strategy::SimulatedAnnealing),
            ("ep", Strategy::ExtremePoint),
            ("extremepoint", Strategy::ExtremePoint),
            ("unknown", Strategy::BottomLeftFill), // Default fallback
        ];

        for (name, expected) in strategies {
            let request = Some(ConfigRequest {
                strategy: Some(name.to_string()),
                spacing: None,
                margin: None,
                time_limit_ms: None,
                target_utilization: None,
                population_size: None,
                max_generations: None,
                crossover_rate: None,
                mutation_rate: None,
            });
            let config = build_config(request);
            assert!(
                std::mem::discriminant(&config.strategy) == std::mem::discriminant(&expected),
                "Strategy '{}' should map to {:?}",
                name,
                expected
            );
        }
    }
}
