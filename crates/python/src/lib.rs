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
    from_json, ConfigRequest, RefusalDetails, Request2D, Request3D, StrategyCatalog,
};
use u_nesting_core::geometry::Boundary;
use u_nesting_core::solver::{Config, Solver};
use u_nesting_core::Error;
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
    /// Stable name of the reason the solve was refused (`invalid_geometry`,
    /// `duplicate_id`, `parameter_out_of_range`, ...); `None` on success.
    code: Option<String>,
    /// The values behind `code` -- the setting, the geometry id, the
    /// positions; `None` on success.
    details: Option<RefusalDetails>,
}

impl SolveOutput {
    /// The output for a refused problem: the message, its `code` and the
    /// values behind it, a setting named by its path (`config.spacing`, see
    /// [`ConfigRequest::locate`]).
    fn refused(e: &Error) -> Self {
        let e = &ConfigRequest::locate(e);
        SolveOutput {
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
            code: Some(e.code().to_string()),
            details: Some(RefusalDetails::of(e)),
        }
    }

    /// The output as the Python dict a solve returns.
    fn into_py(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let output_json = serde_json::to_string(&self)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
        py.import("json")?.call_method1("loads", (output_json,))
    }

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
///         - strategy: "blf", "nfp", "ga", "brkga", "sa", "gdrr", "alns"
///         - spacing: Minimum distance between two placed geometries
///         - margin: Minimum distance from a geometry to the boundary edge
///         - time_limit_ms: Maximum computation time
///         - And the search settings (population_size, max_generations,
///           crossover_rate, mutation_rate, seed, target_utilization,
///           multi_sheet)
///
/// Returns:
///     Dictionary with keys: success, placements, boundaries_used,
///     utilization, total_requested, unplaced, unplaced_count, all_placed,
///     used_bounding_box, used_utilization, computation_time_ms, error, code,
///     details. Every refusal -- a malformed or unknown key, a setting out of
///     range, a geometry or boundary the solver cannot use -- comes back here
///     with `success` False; nothing is raised.
#[pyfunction]
#[pyo3(signature = (geometries, boundary, config=None))]
fn solve_2d<'py>(
    py: Python<'py>,
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let output = match request_json(geometries, boundary, config) {
        Ok(request) => solve_2d_request(&request),
        Err(e) => SolveOutput::refused(&e),
    };
    output.into_py(py)
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
///     config: Optional configuration dictionary (as solve_2d, without
///         multi_sheet)
///
/// Returns:
///     The dictionary solve_2d returns; every refusal comes back in it.
#[pyfunction]
#[pyo3(signature = (geometries, boundary, config=None))]
fn solve_3d<'py>(
    py: Python<'py>,
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let output = match request_json(geometries, boundary, config) {
        Ok(request) => solve_3d_request(&request),
        Err(e) => SolveOutput::refused(&e),
    };
    output.into_py(py)
}

/// How deep a request's values may nest. The deepest a request goes is a
/// hole's vertex coordinate (`geometries[i].holes[j][k][0]`, five levels);
/// the limit stops a value that contains itself.
const MAX_DEPTH: usize = 32;

/// Where a value sits in the request: under a key, or in a list.
enum Slot<'a> {
    /// The value of a key, by its path (`config.spacing`).
    Field(String),
    /// Element `index` of the list at the path (`geometries[0].polygon`).
    Element(&'a str, usize),
}

impl Slot<'_> {
    fn path(&self) -> String {
        match self {
            Slot::Field(path) => path.clone(),
            Slot::Element(list, index) => format!("{list}[{index}]"),
        }
    }

    /// A number at this slot that is not finite: named by the path of the
    /// value, or of the list it is in and its position there.
    fn not_finite(&self, id: Option<&str>, got: f64) -> Error {
        let (parameter, index) = match self {
            Slot::Field(path) => (path.clone(), None),
            Slot::Element(list, index) => (list.to_string(), Some(*index)),
        };
        Error::NotFinite {
            parameter,
            id: id.map(str::to_string),
            index,
            got,
        }
    }
}

/// The request a solve's arguments make, as JSON text the engine's request
/// reader takes -- the same reader, and so the same refusals, as the C ABI
/// and WebAssembly requests. A value JSON cannot hold is refused here, named
/// by its path: a number that is not finite (`value_not_finite`), or an
/// object that is not a JSON value (`malformed_input`).
fn request_json(
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> Result<String, Error> {
    let mut request = serde_json::Map::new();
    let parts = [
        ("geometries", Some(geometries)),
        ("boundary", Some(boundary)),
        ("config", config),
    ];
    for (key, value) in parts {
        if let Some(value) = value {
            let value = to_json(value, &Slot::Field(key.to_string()), None, 0)?;
            request.insert(key.to_string(), value);
        }
    }
    Ok(serde_json::Value::Object(request).to_string())
}

/// `value` as a JSON value, refusing what JSON cannot hold. `id` is the id of
/// the geometry the value belongs to, when it belongs to one.
fn to_json(
    value: &Bound<'_, PyAny>,
    slot: &Slot<'_>,
    id: Option<&str>,
    depth: usize,
) -> Result<serde_json::Value, Error> {
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
    use serde_json::Value;

    let not_json = |what: String| {
        let path = slot.path();
        Error::malformed(Some(&path), format!("{path}: {what}"))
    };
    if depth > MAX_DEPTH {
        return Err(not_json(format!("nested deeper than {MAX_DEPTH} levels")));
    }
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(n) = value.extract::<i64>() {
            return Ok(Value::from(n));
        }
        if let Ok(n) = value.extract::<u64>() {
            return Ok(Value::from(n));
        }
        // Beyond 64 bits is beyond every range the engine checks; as a float
        // it reaches that check, as the JSON text of it would.
        return value
            .extract::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .ok_or_else(|| not_json("an integer too large for a number".to_string()));
    }
    if value.is_instance_of::<PyFloat>() {
        let got: f64 = value
            .extract()
            .map_err(|e| not_json(format!("not a number: {e}")))?;
        return serde_json::Number::from_f64(got)
            .map(Value::Number)
            .ok_or_else(|| slot.not_finite(id, got));
    }
    if value.is_instance_of::<PyString>() {
        let s: String = value
            .extract()
            .map_err(|e| not_json(format!("not a readable string: {e}")))?;
        return Ok(Value::String(s));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let path = slot.path();
        let mut items = Vec::new();
        let iter = value
            .try_iter()
            .map_err(|e| not_json(format!("not iterable: {e}")))?;
        for (index, item) in iter.enumerate() {
            let item = item.map_err(|e| not_json(format!("not iterable: {e}")))?;
            // A geometry's values name it: `geometries[2].quantity` belongs
            // to the geometry with the id given there.
            let own_id = if path == "geometries" {
                item.cast::<PyDict>()
                    .ok()
                    .and_then(|d| d.get_item("id").ok().flatten())
                    .and_then(|i| i.extract::<String>().ok())
            } else {
                None
            };
            let id = own_id.as_deref().or(id);
            let element = Slot::Element(&path, index);
            items.push(to_json(&item, &element, id, depth + 1)?);
        }
        return Ok(Value::Array(items));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let path = slot.path();
        let mut map = serde_json::Map::new();
        for (key, item) in dict.iter() {
            let key: String = key
                .extract()
                .map_err(|_| not_json(format!("the key {key} is not a string")))?;
            let field = Slot::Field(format!("{path}.{key}"));
            let item = to_json(&item, &field, id, depth + 1)?;
            map.insert(key, item);
        }
        return Ok(Value::Object(map));
    }
    let kind = value
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "value".to_string());
    Err(not_json(format!("a {kind} is not a JSON value")))
}

/// The output for a solved layout.
fn solved_output(result: u_nesting_core::SolveResult<f64>) -> SolveOutput {
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
        code: None,
        details: None,
    }
}

/// A 2D request, read and solved; every refusal is the output's.
fn solve_2d_request(json: &str) -> SolveOutput {
    let solve = || -> u_nesting_core::Result<u_nesting_core::SolveResult<f64>> {
        let request: Request2D = from_json(json)?;
        let geometries = Geometry2D::from_requests(request.geometries)?;
        // Exactly one shape: width and height together, or a polygon.
        let boundary = Boundary2D::from_request(request.boundary)?;
        let multi_sheet = request
            .config
            .as_ref()
            .and_then(|c| c.multi_sheet)
            .unwrap_or(false);
        let config = request
            .config
            .as_ref()
            .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config)?;
        // `multi_sheet` distributes overflow across additional sheets.
        let nester = Nester2D::new(config);
        if multi_sheet {
            let mut result = nester.solve_multi_strip(&geometries, &boundary)?;
            // Localize global strip coordinates to the per-sheet frame.
            let (b_min, b_max) = boundary.aabb();
            result.to_boundary_local(b_max[0] - b_min[0]);
            Ok(result)
        } else {
            nester.solve(&geometries, &boundary)
        }
    };
    solve().map_or_else(|e| SolveOutput::refused(&e), solved_output)
}

/// A 3D request, read and solved; every refusal is the output's.
fn solve_3d_request(json: &str) -> SolveOutput {
    let solve = || -> u_nesting_core::Result<u_nesting_core::SolveResult<f64>> {
        let request: Request3D = from_json(json)?;
        let geometries = Geometry3D::from_requests(request.geometries)?;
        let boundary = Boundary3D::from_request(request.boundary);
        let config = request
            .config
            .as_ref()
            .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config_3d)?;
        Packer3D::new(config).solve(&geometries, &boundary)
    };
    solve().map_or_else(|e| SolveOutput::refused(&e), solved_output)
}

/// Get the library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The strategies each dimension offers and the settings each reads.
///
/// Returns ``{"2d": [{"name": "blf", "reads": ["spacing", ...]}, ...],
/// "3d": [...]}``. It is the table a solve's refusal of a setting the strategy
/// does not read (``invalid_option``) comes from, so a caller can offer only
/// the settings that change the chosen strategy's run. The exact (MILP)
/// strategies are listed only in a build with the `milp` feature, which the
/// published wheels do not have.
#[pyfunction]
fn available_strategies(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let catalog = serde_json::to_string(&StrategyCatalog::of_build())
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))?;
    py.import("json")?.call_method1("loads", (catalog,))
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

/// Tests for the interpreter-independent half of the binding: reading and
/// solving the request the arguments make, configuration building, and
/// derived output accounting.
///
/// The `#[pyfunction]` entry points add only the conversion of Python values
/// to that request ([`request_json`]), which needs a live interpreter;
/// everything after it is asserted here so a regression in the contract
/// surfaces at `cargo test` rather than at a consumer's first call.
#[cfg(test)]
mod tests {
    use super::*;
    use u_nesting_core::api_types::{
        Boundary2DRequest, Boundary3DRequest, Geometry2DRequest, Geometry3DRequest,
    };
    use u_nesting_core::placement::Placement;
    use u_nesting_core::SolveResult;

    use u_nesting_core::solver::Strategy;

    fn config_input() -> ConfigRequest {
        ConfigRequest::default()
    }

    /// The configuration a solve builds from its `config` argument.
    fn build_config(input: Option<ConfigRequest>) -> u_nesting_core::Result<Config> {
        input
            .as_ref()
            .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config)
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
    fn every_catalogued_strategy_is_accepted_in_its_dimension() {
        // The catalog is what the module publishes to Python callers; each
        // entry must be accepted by the configuration of its dimension, with
        // every setting it says it reads, or the module advertises a run its
        // own solver refuses.
        let catalog = StrategyCatalog::of_build();
        for (infos, to_config) in [
            (
                &catalog.two_d,
                ConfigRequest::to_config as fn(&ConfigRequest) -> _,
            ),
            (&catalog.three_d, ConfigRequest::to_config_3d),
        ] {
            assert!(!infos.is_empty());
            for info in infos {
                let input = ConfigRequest {
                    strategy: Some(info.name.clone()),
                    ..config_input()
                };
                assert!(
                    to_config(&input).is_ok(),
                    "catalogued strategy '{}' was refused",
                    info.name
                );
            }
        }
    }

    #[test]
    fn unknown_strategy_is_rejected_by_name() {
        let input = ConfigRequest {
            strategy: Some("teleport".to_string()),
            ..config_input()
        };
        let err = build_config(Some(input)).expect_err("unknown strategy must not fall back");
        assert!(
            err.to_string().contains("teleport"),
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
    fn target_utilization_outside_unit_range_is_refused() {
        // It used to be clamped into [0, 1] here while the engine's own check
        // refused it -- the binding answered a different question than asked.
        for bad in [1.5, -0.5] {
            let err = build_config(Some(ConfigRequest {
                strategy: Some("ga".to_string()),
                target_utilization: Some(bad),
                ..config_input()
            }))
            .expect_err("out of [0, 1]");
            assert_eq!(err.code(), "parameter_out_of_range");
        }
        let ok = build_config(Some(ConfigRequest {
            strategy: Some("ga".to_string()),
            target_utilization: Some(0.9),
            ..config_input()
        }))
        .expect("inside [0, 1]");
        assert_eq!(ok.target_utilization, Some(0.9));
    }

    #[test]
    fn stochastic_parameters_and_seed_reach_the_config() {
        let input = ConfigRequest {
            strategy: Some("ga".to_string()),
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
        assert_eq!(config.population_size, Some(42));
        assert_eq!(config.max_generations, Some(7));
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

    /// A quantity is read as a number: a negative one reaches the same range
    /// check as 0 and comes back as a refusal in the result, naming the
    /// geometry; a fraction is the wrong shape, named by its path.
    #[test]
    fn counts_are_read_as_given() {
        for (quantity, got) in [("-1", -1.0), ("0", 0.0)] {
            let inputs: Vec<Geometry3DRequest> = from_json(&format!(
                r#"[{{"id":"crate","dimensions":[1,1,1],"quantity":{quantity}}}]"#
            ))
            .expect("a whole number is the right shape");
            let e = Geometry3D::from_requests(inputs).expect_err("below 1");
            let out = serde_json::to_value(SolveOutput::refused(&e)).expect("serialize");
            assert_eq!(out["success"], false);
            assert_eq!(out["code"], "parameter_out_of_range");
            assert_eq!(
                out["details"],
                serde_json::json!({
                    "parameter": "geometries[0].quantity", "id": "crate", "min": 1.0, "got": got
                })
            );
        }

        let e = from_json::<Vec<Geometry2DRequest>>(
            r#"[{"id":"a","polygon":[[0,0],[1,0],[1,1]],"quantity":2.5}]"#,
        )
        .expect_err("a fraction is not a count");
        assert_eq!(e.code(), "malformed_input");
        assert!(
            e.to_string()
                .starts_with("[0].quantity: must be a whole number"),
            "{e}"
        );

        // A budget left out is left to the strategy.
        let omitted = build_config(Some(config_input())).expect("defaults");
        assert_eq!(omitted.population_size, None);
        assert_eq!(omitted.max_generations, None);

        let e = from_json::<ConfigRequest>(r#"{"strategy":"ga","population_size":-1}"#)
            .expect("a whole number is the right shape")
            .to_config()
            .expect_err("below 2");
        assert_eq!(
            e.to_string(),
            "population_size must be a whole number >= 2, got -1"
        );
    }

    /// No geometries, and a boundary side of 0, come back as refusals in the
    /// result dict, naming the field -- the solver's refusals.
    #[test]
    fn the_solver_refuses_an_empty_list_and_a_flat_boundary() {
        let refused = |e: Error| serde_json::to_value(SolveOutput::refused(&e)).expect("serialize");
        let e = Nester2D::new(Config::default())
            .solve(&[], &Boundary2D::rectangle(10.0, 10.0))
            .expect_err("nothing to place");
        let out = refused(e);
        assert_eq!(out["code"], "empty_input");
        assert_eq!(
            out["details"],
            serde_json::json!({ "parameter": "geometries" })
        );
        let e = Packer3D::new(Config::default())
            .solve(&[], &Boundary3D::new(1.0, 1.0, 1.0))
            .expect_err("nothing to pack");
        assert_eq!(refused(e)["code"], "empty_input");

        let flat: Boundary2DRequest =
            from_json(r#"{"width": 0, "height": 10}"#).expect("the right shape");
        let e = Nester2D::new(Config::default())
            .solve(
                &[Geometry2D::rectangle("a", 1.0, 1.0)],
                &Boundary2D::from_request(flat).expect("a rectangle"),
            )
            .expect_err("no width");
        assert_eq!(
            refused(e)["details"],
            serde_json::json!({ "parameter": "boundary.width", "exclusive_min": 0.0, "got": 0.0 })
        );
    }

    /// A setting the solve refuses comes back in the result dict named by its
    /// path in the request -- `config.strategy` -- as the configuration's own
    /// refusals are, while the Rust error keeps the field name.
    #[test]
    fn a_refused_setting_is_named_by_its_path() {
        let e = Packer3D::new(Config::default().with_strategy(Strategy::NfpGuided))
            .solve(
                &[Geometry3D::new("b", 1.0, 1.0, 1.0)],
                &Boundary3D::new(5.0, 5.0, 5.0),
            )
            .expect_err("2D only");
        assert_eq!(e.parameter(), Some("strategy"));
        let out = serde_json::to_value(SolveOutput::refused(&e)).expect("serialize");
        assert_eq!(out["code"], "invalid_option");
        assert_eq!(out["details"]["parameter"], "config.strategy");
        assert!(
            out["error"]
                .as_str()
                .expect("a message")
                .starts_with("config.strategy:"),
            "{out}"
        );

        let e = build_config(Some(ConfigRequest {
            spacing: Some(-1.0),
            ..config_input()
        }))
        .expect_err("negative");
        let out = serde_json::to_value(SolveOutput::refused(&e)).expect("serialize");
        assert_eq!(out["details"]["parameter"], "config.spacing");

        // 3D has no sheets: the 2D setting is refused, not ignored.
        let e = ConfigRequest {
            multi_sheet: Some(true),
            ..config_input()
        }
        .to_config_3d()
        .expect_err("2D only");
        assert_eq!(
            serde_json::to_value(SolveOutput::refused(&e)).expect("serialize")["details"]
                ["parameter"],
            "config.multi_sheet"
        );
    }
    /// Every refusal is in the output -- the configuration's, the
    /// boundary's, the request's shape -- each named by its path in the
    /// request, as the C ABI names it. None of them is raised.
    #[test]
    fn every_refusal_is_in_the_output_named_by_its_path() {
        let plate = r#"[{"id": "p", "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]]}]"#;
        let sheet = r#"{"width": 10, "height": 10}"#;
        let solve = |geometries: &str, boundary: &str, config: &str| {
            let request = format!(
                r#"{{"geometries": {geometries}, "boundary": {boundary}, "config": {config}}}"#
            );
            serde_json::to_value(solve_2d_request(&request)).expect("serialize")
        };
        let cases = [
            (
                solve(plate, sheet, r#"{"strategy": "ga", "population_size": 1}"#),
                "parameter_out_of_range",
                "config.population_size",
            ),
            (
                solve(plate, sheet, r#"{"strategy": "ep"}"#),
                "invalid_option",
                "config.strategy",
            ),
            (
                solve(plate, sheet, r#"{"strategy": "teleport"}"#),
                "unknown_option",
                "config.strategy",
            ),
            (
                solve(plate, sheet, r#"{"seed": 2.5}"#),
                "malformed_input",
                "config.seed",
            ),
            (
                solve(plate, sheet, r#"{"stratgy": "nfp"}"#),
                "malformed_input",
                "config.stratgy",
            ),
            (
                solve(
                    plate,
                    r#"{"width": 10, "height": 10, "polygon": [[0, 0], [1, 0], [1, 1]]}"#,
                    "{}",
                ),
                "invalid_boundary",
                "boundary",
            ),
            (solve("[]", sheet, "{}"), "empty_input", "geometries"),
        ];
        for (out, code, parameter) in cases {
            assert_eq!(out["success"], false, "{out}");
            assert_eq!(out["code"], code, "{out}");
            assert_eq!(out["details"]["parameter"], parameter, "{out}");
            assert!(out["error"].is_string(), "{out}");
        }

        let request = r#"{"geometries": [{"id": "b", "dimensions": [1, 1, 1]}],
                          "boundary": {"dimensions": [5, 5, 5]},
                          "config": {"multi_sheet": false}}"#;
        let out = serde_json::to_value(solve_3d_request(request)).expect("serialize");
        assert_eq!(out["code"], "invalid_option");
        assert_eq!(out["details"]["parameter"], "config.multi_sheet");
    }

    /// A request read whole solves as the C ABI's does.
    #[test]
    fn a_request_solves() {
        let out = solve_2d_request(
            r#"{"geometries": [{"id": "p", "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]], "quantity": 3}],
                "boundary": {"width": 10, "height": 10}}"#,
        );
        assert!(out.success, "{:?}", out.error);
        assert_eq!(out.placements.len(), 3);
        assert_eq!(out.total_requested, 3);
        let out = solve_3d_request(
            r#"{"geometries": [{"id": "b", "dimensions": [1, 2, 3], "quantity": 2}],
                "boundary": {"dimensions": [5, 5, 5]}, "config": {"strategy": "ep"}}"#,
        );
        assert!(out.success, "{:?}", out.error);
        assert_eq!(out.placements.len(), 2);
    }

    /// A number that is not finite is named where it sits: by the path of
    /// the value, or of the list and its position there, with the geometry
    /// it belongs to.
    #[test]
    fn a_number_that_is_not_finite_is_named_where_it_sits() {
        let refused = |e: Error| serde_json::to_value(SolveOutput::refused(&e)).expect("serialize");
        let out = refused(Slot::Field("config.spacing".into()).not_finite(None, f64::NAN));
        assert_eq!(out["code"], "value_not_finite");
        assert_eq!(
            out["details"],
            serde_json::json!({ "parameter": "config.spacing" })
        );
        assert_eq!(
            out["error"],
            "config.spacing: expected a finite number, got NaN"
        );

        let out = refused(
            Slot::Element("geometries[0].polygon[2]", 1).not_finite(Some("p"), f64::INFINITY),
        );
        assert_eq!(
            out["details"],
            serde_json::json!({ "parameter": "geometries[0].polygon[2]", "id": "p", "index": 1 })
        );
        assert_eq!(
            out["error"],
            "geometries[0].polygon[2][1]: expected a finite number, got inf"
        );
    }
}
