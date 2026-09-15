//! Python bindings for U-Nesting.
//!
//! This crate provides Python bindings using PyO3 for the U-Nesting
//! 2D nesting and 3D bin packing engine.
//!
//! ## Installation
//!
//! ```bash
//! pip install u-nesting
//! ```
//!
//! ## Usage
//!
//! ```python
//! import u_nesting
//!
//! # 2D Nesting
//! result = u_nesting.solve_2d(
//!     geometries=[
//!         {"id": "rect1", "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]], "quantity": 5}
//!     ],
//!     boundary={"width": 500, "height": 300},
//!     config={"strategy": "nfp", "spacing": 2.0}
//! )
//!
//! # 3D Bin Packing
//! result = u_nesting.solve_3d(
//!     geometries=[
//!         {"id": "box1", "dimensions": [100, 50, 30], "quantity": 10}
//!     ],
//!     boundary={"dimensions": [500, 400, 300]},
//!     config={"strategy": "ep"}
//! )
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use u_nesting_core::api_types::{
    Boundary2DRequest, Boundary3DRequest, ConfigRequest, Geometry2DRequest, Geometry3DRequest,
};
use u_nesting_core::geometry::Boundary;
use u_nesting_core::solver::{Config, Solver, Strategy};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};

// Inputs use the canonical request types from `u-nesting-core::api_types`, the
// same contract the C and WebAssembly bindings deserialize. Sharing them means a
// field added to the engine's input surface reaches every binding at once, and
// that an unknown key (a misspelled `quantity`, say) is rejected here exactly as
// it is elsewhere rather than silently falling back to a default.
//
// Outputs deliberately do **not** follow: the response types below are
// dimension-agnostic (`position`/`rotation` are vectors, so one shape serves 2D
// and 3D), whereas the canonical responses are split into flat 2D and 3D
// variants with different key names. Unifying them would rename keys in a
// published API, which is a decision for a release boundary, not a refactor.

/// Placement output.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlacementOutput {
    geometry_id: String,
    instance: usize,
    position: Vec<f64>,
    rotation: Vec<f64>,
    boundary_index: usize,
}

/// Solve result output.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SolveOutput {
    success: bool,
    placements: Vec<PlacementOutput>,
    boundaries_used: usize,
    utilization: f64,
    /// Total geometry instances requested (Σ quantity). `placements` is
    /// instance-level while `unplaced` is deduplicated geometry IDs; the
    /// instance-level unplaced count is `total_requested - len(placements)`.
    #[serde(default)]
    total_requested: usize,
    unplaced: Vec<String>,
    /// Instance-level count of unplaced instances (`total_requested - len(placements)`).
    /// Satisfies `len(placements) + unplaced_count == total_requested`. Unlike the
    /// deduplicated `unplaced` ID list, this never undercounts a multi-quantity geometry.
    #[serde(default)]
    unplaced_count: usize,
    /// Whether every requested instance was placed. Prefer this over `success`
    /// (which only reports that the solve completed without error) to detect
    /// partial packing.
    #[serde(default)]
    all_placed: bool,
    /// Used-footprint bounding box `[width, height]` (2D only; `[0, 0]` for 3D).
    /// Boundary-padding independent, unlike `utilization`.
    #[serde(default)]
    used_bounding_box: [f64; 2],
    /// Packing-density metric (2D only; `0` for 3D): `piece_area / (used_w × used_h)`
    /// against the used bounding box. The denominator shrinks on **both** axes, so
    /// this is **not** a fixed-width stock-efficiency figure — for fixed-width stock
    /// (fabric/coil/sheet) the unused width is real waste. Fixed-width consumers
    /// should compute `piece_area / (boundary_width × used_bounding_box[1])`.
    #[serde(default)]
    used_utilization: f64,
    computation_time_ms: u64,
    error: Option<String>,
}

impl SolveOutput {
    /// Fills the derived accounting/metric fields from a solved result, applying
    /// the same instance-level invariant as the FFI/WASM `From` impl.
    fn accounting(result: &u_nesting_core::SolveResult<f64>) -> (usize, bool, [f64; 2], f64) {
        let unplaced_count = result
            .total_requested
            .saturating_sub(result.placements.len());
        let [uw, uh] = result.used_bounding_box;
        let used_area = uw * uh;
        let used_utilization = if used_area > 0.0 {
            result.total_piece_area / used_area
        } else {
            0.0
        };
        (
            unplaced_count,
            unplaced_count == 0,
            result.used_bounding_box,
            used_utilization,
        )
    }
}

/// Builds a solver [`Config`] from Python input, validating each field.
///
/// Returns `Err(message)` on invalid input (negative/non-finite spacing or
/// margin, unknown strategy name) so the caller raises `ValueError` instead of
/// silently applying a default. Strategy names are parsed by the canonical
/// [`Strategy::parse`], the single source of truth shared across all bindings.
fn build_config(input: Option<ConfigRequest>) -> Result<Config, String> {
    let mut config = Config::default();

    if let Some(c) = input {
        if let Some(strategy) = c.strategy {
            config.strategy = Strategy::parse(&strategy)
                .ok_or_else(|| format!("unknown strategy: '{strategy}'"))?;
        }
        if let Some(spacing) = c.spacing {
            if !spacing.is_finite() || spacing < 0.0 {
                return Err(format!(
                    "spacing must be a non-negative, finite number (got {spacing})"
                ));
            }
            config.spacing = spacing;
        }
        if let Some(margin) = c.margin {
            if !margin.is_finite() || margin < 0.0 {
                return Err(format!(
                    "margin must be a non-negative, finite number (got {margin})"
                ));
            }
            config.margin = margin;
        }
        if let Some(time_limit) = c.time_limit_ms {
            config.time_limit_ms = time_limit;
        }
        if let Some(target) = c.target_utilization {
            config.target_utilization = Some(target.clamp(0.0, 1.0));
        }
        if let Some(pop) = c.population_size {
            config.population_size = pop;
        }
        if let Some(gens) = c.max_generations {
            config.max_generations = gens;
        }
        if let Some(crossover) = c.crossover_rate {
            config.crossover_rate = crossover;
        }
        if let Some(mutation) = c.mutation_rate {
            config.mutation_rate = mutation;
        }
        if let Some(seed) = c.seed {
            config.seed = Some(seed);
        }
    }

    Ok(config)
}

/// Solve a 2D nesting problem.
///
/// Args:
///     geometries: List of geometry dictionaries with keys:
///         - id (str): Unique identifier
///         - polygon (list): List of [x, y] vertices
///         - quantity (int, optional): Number of copies (default: 1)
///         - rotations (list, optional): Allowed rotation angles in degrees
///         - allow_flip (bool, optional): Allow mirroring (default: False)
///         - holes (list, optional): List of hole polygons
///     boundary: Boundary dictionary with either:
///         - width, height: For rectangular boundary
///         - polygon: For arbitrary boundary shape
///     config: Optional configuration dictionary with keys:
///         - strategy: "blf", "nfp", "ga", "brkga", "sa"
///         - spacing: Minimum distance between two placed geometries
///         - margin: Minimum distance from a geometry to the boundary edge
///         - time_limit_ms: Maximum computation time
///         - And GA-specific parameters
///
/// Returns:
///     Dictionary with keys: success, placements, boundaries_used,
///     utilization, unplaced, computation_time_ms, error
#[pyfunction]
#[pyo3(signature = (geometries, boundary, config=None))]
fn solve_2d<'py>(
    py: Python<'py>,
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Parse geometries
    let geom_json: String = py
        .import("json")?
        .call_method1("dumps", (geometries,))?
        .extract()?;
    let geom_inputs: Vec<Geometry2DRequest> = serde_json::from_str(&geom_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid geometries: {}", e)))?;

    // Parse boundary
    let boundary_json: String = py
        .import("json")?
        .call_method1("dumps", (boundary,))?
        .extract()?;
    let boundary_input: Boundary2DRequest = serde_json::from_str(&boundary_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid boundary: {}", e)))?;

    // Parse config
    let config_input: Option<ConfigRequest> = if let Some(cfg) = config {
        let cfg_json: String = py
            .import("json")?
            .call_method1("dumps", (cfg,))?
            .extract()?;
        Some(
            serde_json::from_str(&cfg_json)
                .map_err(|e| PyValueError::new_err(format!("Invalid config: {}", e)))?,
        )
    } else {
        None
    };

    // Convert to internal types
    let rust_geometries: Vec<Geometry2D> = geom_inputs
        .into_iter()
        .map(|g| {
            let vertices: Vec<(f64, f64)> = g.polygon.into_iter().map(|p| (p[0], p[1])).collect();
            let mut geom = Geometry2D::new(&g.id)
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

    let rust_boundary = if let (Some(w), Some(h)) = (boundary_input.width, boundary_input.height) {
        Boundary2D::rectangle(w, h)
    } else if let Some(polygon) = boundary_input.polygon {
        let vertices: Vec<(f64, f64)> = polygon.into_iter().map(|p| (p[0], p[1])).collect();
        Boundary2D::new(vertices)
    } else {
        return Err(PyValueError::new_err(
            "Boundary must have width/height or polygon",
        ));
    };

    // Read the multi-sheet flag before `build_config` consumes the config.
    let multi_sheet = config_input
        .as_ref()
        .and_then(|c| c.multi_sheet)
        .unwrap_or(false);

    let rust_config = build_config(config_input)
        .map_err(|e| PyValueError::new_err(format!("Invalid config: {e}")))?;

    // Solve — `multi_sheet` distributes overflow across additional sheets.
    let nester = Nester2D::new(rust_config);
    let solved = if multi_sheet {
        nester.solve_multi_strip(&rust_geometries, &rust_boundary)
    } else {
        nester.solve(&rust_geometries, &rust_boundary)
    };
    let output = match solved {
        Ok(mut result) => {
            if multi_sheet {
                // Localize global strip coordinates to the per-sheet frame.
                let (b_min, b_max) = rust_boundary.aabb();
                result.to_boundary_local(b_max[0] - b_min[0]);
            }
            let (unplaced_count, all_placed, used_bounding_box, used_utilization) =
                SolveOutput::accounting(&result);
            SolveOutput {
                success: true,
                placements: result
                    .placements
                    .into_iter()
                    .map(|p| PlacementOutput {
                        geometry_id: p.geometry_id,
                        instance: p.instance,
                        position: p.position,
                        rotation: p.rotation,
                        boundary_index: p.boundary_index,
                    })
                    .collect(),
                boundaries_used: result.boundaries_used,
                utilization: result.utilization,
                total_requested: result.total_requested,
                unplaced: result.unplaced,
                unplaced_count,
                all_placed,
                used_bounding_box,
                used_utilization,
                computation_time_ms: result.computation_time_ms,
                error: None,
            }
        }
        Err(e) => SolveOutput {
            success: false,
            placements: vec![],
            boundaries_used: 0,
            utilization: 0.0,
            total_requested: 0,
            unplaced: vec![],
            unplaced_count: 0,
            all_placed: false,
            used_bounding_box: [0.0, 0.0],
            used_utilization: 0.0,
            computation_time_ms: 0,
            error: Some(e.to_string()),
        },
    };

    // Convert to Python dict
    let output_json = serde_json::to_string(&output)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
    let json_module = py.import("json")?;
    let result = json_module.call_method1("loads", (output_json,))?;
    Ok(result)
}

/// Solve a 3D bin packing problem.
///
/// Args:
///     geometries: List of geometry dictionaries with keys:
///         - id (str): Unique identifier
///         - dimensions (list): [width, depth, height]
///         - quantity (int, optional): Number of copies (default: 1)
///         - mass (float, optional): Item mass
///         - orientation (str, optional): "any", "upright", or "fixed"
///     boundary: Boundary dictionary with keys:
///         - dimensions (list): [width, depth, height]
///         - max_mass (float, optional): Maximum total mass
///         - gravity (bool, optional): Enable gravity constraints
///         - stability (bool, optional): Enable stability constraints
///     config: Optional configuration dictionary (same as solve_2d)
///
/// Returns:
///     Dictionary with keys: success, placements, boundaries_used,
///     utilization, unplaced, computation_time_ms, error
#[pyfunction]
#[pyo3(signature = (geometries, boundary, config=None))]
fn solve_3d<'py>(
    py: Python<'py>,
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Parse geometries
    let geom_json: String = py
        .import("json")?
        .call_method1("dumps", (geometries,))?
        .extract()?;
    let geom_inputs: Vec<Geometry3DRequest> = serde_json::from_str(&geom_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid geometries: {}", e)))?;

    // Parse boundary
    let boundary_json: String = py
        .import("json")?
        .call_method1("dumps", (boundary,))?
        .extract()?;
    let boundary_input: Boundary3DRequest = serde_json::from_str(&boundary_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid boundary: {}", e)))?;

    // Parse config
    let config_input: Option<ConfigRequest> = if let Some(cfg) = config {
        let cfg_json: String = py
            .import("json")?
            .call_method1("dumps", (cfg,))?
            .extract()?;
        Some(
            serde_json::from_str(&cfg_json)
                .map_err(|e| PyValueError::new_err(format!("Invalid config: {}", e)))?,
        )
    } else {
        None
    };

    // Convert to internal types
    let rust_geometries: Vec<Geometry3D> = geom_inputs
        .into_iter()
        .map(|g| {
            let mut geom =
                Geometry3D::new(&g.id, g.dimensions[0], g.dimensions[1], g.dimensions[2])
                    .with_quantity(g.quantity);

            if let Some(mass) = g.mass {
                geom = geom.with_mass(mass);
            }

            geom
        })
        .collect();

    let mut rust_boundary = Boundary3D::new(
        boundary_input.dimensions[0],
        boundary_input.dimensions[1],
        boundary_input.dimensions[2],
    );

    if let Some(max_mass) = boundary_input.max_mass {
        rust_boundary = rust_boundary.with_max_mass(max_mass);
    }

    rust_boundary = rust_boundary
        .with_gravity(boundary_input.gravity)
        .with_stability(boundary_input.stability);

    let rust_config = build_config(config_input)
        .map_err(|e| PyValueError::new_err(format!("Invalid config: {e}")))?;

    // Solve
    let packer = Packer3D::new(rust_config);
    let output = match packer.solve(&rust_geometries, &rust_boundary) {
        Ok(result) => {
            let (unplaced_count, all_placed, used_bounding_box, used_utilization) =
                SolveOutput::accounting(&result);
            SolveOutput {
                success: true,
                placements: result
                    .placements
                    .into_iter()
                    .map(|p| PlacementOutput {
                        geometry_id: p.geometry_id,
                        instance: p.instance,
                        position: p.position,
                        rotation: p.rotation,
                        boundary_index: p.boundary_index,
                    })
                    .collect(),
                boundaries_used: result.boundaries_used,
                utilization: result.utilization,
                total_requested: result.total_requested,
                unplaced: result.unplaced,
                unplaced_count,
                all_placed,
                used_bounding_box,
                used_utilization,
                computation_time_ms: result.computation_time_ms,
                error: None,
            }
        }
        Err(e) => SolveOutput {
            success: false,
            placements: vec![],
            boundaries_used: 0,
            utilization: 0.0,
            total_requested: 0,
            unplaced: vec![],
            unplaced_count: 0,
            all_placed: false,
            used_bounding_box: [0.0, 0.0],
            used_utilization: 0.0,
            computation_time_ms: 0,
            error: Some(e.to_string()),
        },
    };

    // Convert to Python dict
    let output_json = serde_json::to_string(&output)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
    let json_module = py.import("json")?;
    let result = json_module.call_method1("loads", (output_json,))?;
    Ok(result)
}

/// Get the library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// List available strategies.
///
/// The union of the 2D and 3D strategy names this build accepts: `nfp`, `gdrr`
/// and `alns` apply to 2D nesting, `ep` to 3D packing, and the rest to both.
/// Exact (MILP) strategies are deliberately absent — the public path currently
/// handles axis-aligned rectangles only, so advertising them would promise more
/// than it delivers.
#[pyfunction]
fn available_strategies() -> Vec<&'static str> {
    vec!["blf", "nfp", "ga", "brkga", "sa", "ep", "gdrr", "alns"]
}

/// U-Nesting Python module.
#[pymodule]
fn u_nesting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_2d, m)?)?;
    m.add_function(wrap_pyfunction!(solve_3d, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available_strategies, m)?)?;
    Ok(())
}

/// Tests for the interpreter-independent half of the binding: input
/// deserialization, configuration building, and derived output accounting.
///
/// The `#[pyfunction]` entry points themselves need a live interpreter and are
/// exercised end-to-end by the packaging pipeline instead; everything reachable
/// without one is asserted here so a regression in the contract surfaces at
/// `cargo test` rather than at a consumer's first call.
#[cfg(test)]
mod tests {
    use super::*;
    use u_nesting_core::placement::Placement;
    use u_nesting_core::SolveResult;

    fn config_input() -> ConfigRequest {
        ConfigRequest::default()
    }

    // ---- build_config: strategy ----

    #[test]
    fn none_input_yields_engine_defaults() {
        let config = build_config(None).expect("default config is always valid");
        let expected = Config::default();
        assert_eq!(config.strategy, expected.strategy);
        assert_eq!(config.spacing, expected.spacing);
        assert_eq!(config.margin, expected.margin);
        assert_eq!(config.time_limit_ms, expected.time_limit_ms);
        assert_eq!(config.seed, expected.seed);
    }

    #[test]
    fn every_advertised_strategy_is_accepted() {
        // `available_strategies()` is the list the binding publishes to Python
        // callers; each entry must round-trip through the canonical parser, or
        // the module advertises a name its own solver rejects.
        for name in available_strategies() {
            let input = ConfigRequest {
                strategy: Some(name.to_string()),
                ..config_input()
            };
            assert!(
                build_config(Some(input)).is_ok(),
                "advertised strategy '{name}' was rejected by build_config"
            );
        }
    }

    #[test]
    fn advertised_strategies_cover_every_supported_solver() {
        // Regression guard: `gdrr` and `alns` solve 2D problems in this build but
        // were missing from the published list, so callers could not discover
        // them. Locked as a golden set — extending the solvers means extending
        // this list in the same change.
        assert_eq!(
            available_strategies(),
            vec!["blf", "nfp", "ga", "brkga", "sa", "ep", "gdrr", "alns"]
        );
    }

    #[test]
    fn unknown_strategy_is_rejected_by_name() {
        let input = ConfigRequest {
            strategy: Some("teleport".to_string()),
            ..config_input()
        };
        let err = build_config(Some(input)).expect_err("unknown strategy must not fall back");
        assert!(
            err.contains("teleport"),
            "error should name the offending strategy, got: {err}"
        );
    }

    #[test]
    fn strategy_parsing_tolerates_case_and_padding() {
        let input = ConfigRequest {
            strategy: Some("  NFP ".to_string()),
            ..config_input()
        };
        let config = build_config(Some(input)).expect("canonical parser trims and lowercases");
        assert_eq!(config.strategy, Strategy::NfpGuided);
    }

    // ---- build_config: numeric validation ----

    #[test]
    fn spacing_rejects_negative_and_non_finite() {
        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let input = ConfigRequest {
                spacing: Some(bad),
                ..config_input()
            };
            assert!(
                build_config(Some(input)).is_err(),
                "spacing {bad} must be rejected rather than silently applied"
            );
        }
    }

    #[test]
    fn margin_rejects_negative_and_non_finite() {
        for bad in [-0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let input = ConfigRequest {
                margin: Some(bad),
                ..config_input()
            };
            assert!(
                build_config(Some(input)).is_err(),
                "margin {bad} must be rejected rather than silently applied"
            );
        }
    }

    #[test]
    fn zero_spacing_and_margin_are_valid() {
        let input = ConfigRequest {
            spacing: Some(0.0),
            margin: Some(0.0),
            ..config_input()
        };
        let config = build_config(Some(input)).expect("zero is the boundary value, not an error");
        assert_eq!(config.spacing, 0.0);
        assert_eq!(config.margin, 0.0);
    }

    #[test]
    fn target_utilization_is_clamped_to_unit_range() {
        let over = build_config(Some(ConfigRequest {
            target_utilization: Some(1.5),
            ..config_input()
        }))
        .expect("out-of-range targets clamp instead of erroring");
        assert_eq!(over.target_utilization, Some(1.0));

        let under = build_config(Some(ConfigRequest {
            target_utilization: Some(-0.5),
            ..config_input()
        }))
        .expect("out-of-range targets clamp instead of erroring");
        assert_eq!(under.target_utilization, Some(0.0));
    }

    #[test]
    fn stochastic_parameters_and_seed_reach_the_config() {
        let input = ConfigRequest {
            time_limit_ms: Some(1_234),
            population_size: Some(42),
            max_generations: Some(7),
            crossover_rate: Some(0.5),
            mutation_rate: Some(0.25),
            seed: Some(99),
            ..config_input()
        };
        let config = build_config(Some(input)).expect("valid tuning parameters");
        assert_eq!(config.time_limit_ms, 1_234);
        assert_eq!(config.population_size, 42);
        assert_eq!(config.max_generations, 7);
        assert_eq!(config.crossover_rate, 0.5);
        assert_eq!(config.mutation_rate, 0.25);
        assert_eq!(config.seed, Some(99));
    }

    // ---- SolveOutput::accounting ----

    fn result_with(placed: usize, requested: usize) -> SolveResult<f64> {
        let mut result = SolveResult::new();
        result.placements = (0..placed)
            .map(|i| Placement::new_2d("part".to_string(), i, 0.0, 0.0, 0.0))
            .collect();
        result.total_requested = requested;
        result
    }

    #[test]
    fn partial_packing_reports_instance_level_shortfall() {
        let (unplaced_count, all_placed, _, _) = SolveOutput::accounting(&result_with(3, 5));
        assert_eq!(unplaced_count, 2);
        assert!(!all_placed);
    }

    #[test]
    fn complete_packing_reports_all_placed() {
        let (unplaced_count, all_placed, _, _) = SolveOutput::accounting(&result_with(5, 5));
        assert_eq!(unplaced_count, 0);
        assert!(all_placed);
    }

    #[test]
    fn shortfall_saturates_instead_of_underflowing() {
        // `total_requested` is set at the top-level entry point; a solver path
        // that ever reports more placements than requested must not wrap around.
        let (unplaced_count, all_placed, _, _) = SolveOutput::accounting(&result_with(4, 0));
        assert_eq!(unplaced_count, 0);
        assert!(all_placed);
    }

    #[test]
    fn used_utilization_guards_against_empty_footprint() {
        let mut result = result_with(0, 2);
        result.total_piece_area = 100.0;
        result.used_bounding_box = [0.0, 0.0];
        let (_, _, bbox, used_utilization) = SolveOutput::accounting(&result);
        assert_eq!(bbox, [0.0, 0.0]);
        assert_eq!(used_utilization, 0.0, "must not divide by a zero footprint");
    }

    #[test]
    fn used_utilization_is_area_over_used_bounding_box() {
        let mut result = result_with(1, 1);
        result.total_piece_area = 50.0;
        result.used_bounding_box = [10.0, 20.0];
        let (_, _, bbox, used_utilization) = SolveOutput::accounting(&result);
        assert_eq!(bbox, [10.0, 20.0]);
        assert_eq!(used_utilization, 0.25);
    }

    // ---- input deserialization defaults ----

    #[test]
    fn geometry_2d_input_defaults_to_a_single_unflipped_copy() {
        let input: Geometry2DRequest =
            serde_json::from_str(r#"{"id":"a","polygon":[[0,0],[1,0],[1,1]]}"#)
                .expect("quantity/rotations/holes/allow_flip are all optional");
        assert_eq!(input.quantity, 1);
        assert!(!input.allow_flip);
        assert!(input.holes.is_none());
        assert!(input.rotations.is_none());
    }

    #[test]
    fn geometry_3d_input_defaults_to_a_single_copy() {
        let input: Geometry3DRequest = serde_json::from_str(r#"{"id":"b","dimensions":[1,2,3]}"#)
            .expect("quantity/mass/orientation are all optional");
        assert_eq!(input.quantity, 1);
        assert!(input.mass.is_none());
        assert!(input.orientation.is_none());
    }

    #[test]
    fn boundary_3d_input_defaults_to_no_physics() {
        let input: Boundary3DRequest =
            serde_json::from_str(r#"{"dimensions":[10,10,10]}"#).expect("physics flags default");
        assert!(!input.gravity);
        assert!(!input.stability);
        assert!(input.max_mass.is_none());
    }

    #[test]
    fn empty_config_object_is_accepted() {
        let input: ConfigRequest =
            serde_json::from_str("{}").expect("every config key is optional");
        assert!(build_config(Some(input)).is_ok());
    }

    // ---- strict input contract (shared canonical request types) ----

    #[test]
    fn misspelled_geometry_key_is_rejected() {
        // Silently defaulting a misspelled `quantity` to 1 is the worst failure
        // shape available: the solve succeeds and quietly places the wrong number
        // of parts. The canonical request types reject unknown keys instead.
        let err = serde_json::from_str::<Geometry2DRequest>(
            r#"{"id":"a","polygon":[[0,0],[1,0],[1,1]],"quantiy":5}"#,
        )
        .expect_err("unknown geometry key must be rejected");
        assert!(
            err.to_string().contains("quantiy"),
            "error should name the unknown key, got: {err}"
        );
    }

    #[test]
    fn misspelled_config_key_is_rejected() {
        let err = serde_json::from_str::<ConfigRequest>(r#"{"stratgy":"nfp"}"#)
            .expect_err("unknown config key must be rejected");
        assert!(
            err.to_string().contains("stratgy"),
            "error should name the unknown key, got: {err}"
        );
    }

    #[test]
    fn misspelled_boundary_key_is_rejected() {
        assert!(
            serde_json::from_str::<Boundary2DRequest>(r#"{"width":10,"hieght":20}"#).is_err(),
            "unknown boundary key must be rejected"
        );
        assert!(
            serde_json::from_str::<Boundary3DRequest>(r#"{"dimensions":[1,2,3],"gravty":true}"#)
                .is_err(),
            "unknown 3D boundary key must be rejected"
        );
    }

    #[test]
    fn multi_sheet_survives_on_the_shared_config_type() {
        // `multi_sheet` is read directly off the request in `solve_2d` rather than
        // through `build_config`, so the shared type must still carry it.
        let input: ConfigRequest =
            serde_json::from_str(r#"{"multi_sheet":true}"#).expect("multi_sheet is a known key");
        assert_eq!(input.multi_sheet, Some(true));
    }
}
