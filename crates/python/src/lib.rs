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

use u_nesting_core::solver::{Config, Solver, Strategy};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};

/// 2D Geometry input.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Geometry2DInput {
    id: String,
    polygon: Vec<[f64; 2]>,
    #[serde(default)]
    holes: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(default = "default_quantity")]
    quantity: usize,
    #[serde(default)]
    rotations: Option<Vec<f64>>,
    #[serde(default)]
    allow_flip: bool,
}

/// 2D Boundary input.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Boundary2DInput {
    width: Option<f64>,
    height: Option<f64>,
    polygon: Option<Vec<[f64; 2]>>,
}

/// 3D Geometry input.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Geometry3DInput {
    id: String,
    dimensions: [f64; 3],
    #[serde(default = "default_quantity")]
    quantity: usize,
    #[serde(default)]
    mass: Option<f64>,
    #[serde(default)]
    orientation: Option<String>,
}

/// 3D Boundary input.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Boundary3DInput {
    dimensions: [f64; 3],
    #[serde(default)]
    max_mass: Option<f64>,
    #[serde(default)]
    gravity: bool,
    #[serde(default)]
    stability: bool,
}

/// Config input.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ConfigInput {
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    spacing: Option<f64>,
    #[serde(default)]
    margin: Option<f64>,
    #[serde(default)]
    time_limit_ms: Option<u64>,
    #[serde(default)]
    target_utilization: Option<f64>,
    #[serde(default)]
    population_size: Option<usize>,
    #[serde(default)]
    max_generations: Option<u32>,
    #[serde(default)]
    crossover_rate: Option<f64>,
    #[serde(default)]
    mutation_rate: Option<f64>,
}

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
    unplaced: Vec<String>,
    computation_time_ms: u64,
    error: Option<String>,
}

fn default_quantity() -> usize {
    1
}

fn parse_strategy(s: &str) -> Strategy {
    match s.to_lowercase().as_str() {
        "blf" | "bottomleftfill" => Strategy::BottomLeftFill,
        "nfp" | "nfpguided" => Strategy::NfpGuided,
        "ga" | "genetic" | "geneticalgorithm" => Strategy::GeneticAlgorithm,
        "brkga" => Strategy::Brkga,
        "sa" | "simulatedannealing" => Strategy::SimulatedAnnealing,
        "ep" | "extremepoint" => Strategy::ExtremePoint,
        _ => Strategy::BottomLeftFill,
    }
}

fn build_config(input: Option<ConfigInput>) -> Config {
    let mut config = Config::default();

    if let Some(c) = input {
        if let Some(strategy) = c.strategy {
            config.strategy = parse_strategy(&strategy);
        }
        if let Some(spacing) = c.spacing {
            config.spacing = spacing;
        }
        if let Some(margin) = c.margin {
            config.margin = margin;
        }
        if let Some(time_limit) = c.time_limit_ms {
            config.time_limit_ms = time_limit;
        }
        if let Some(target) = c.target_utilization {
            config.target_utilization = Some(target);
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
    }

    config
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
///         - spacing: Minimum spacing between geometries
///         - margin: Margin from boundary edges
///         - time_limit_ms: Maximum computation time
///         - And GA-specific parameters
///
/// Returns:
///     Dictionary with keys: success, placements, boundaries_used,
///     utilization, unplaced, computation_time_ms, error
#[pyfunction]
#[pyo3(signature = (geometries, boundary, config=None))]
fn solve_2d(
    py: Python<'_>,
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    // Parse geometries
    let geom_json: String = py
        .import_bound("json")?
        .call_method1("dumps", (geometries,))?
        .extract()?;
    let geom_inputs: Vec<Geometry2DInput> = serde_json::from_str(&geom_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid geometries: {}", e)))?;

    // Parse boundary
    let boundary_json: String = py
        .import_bound("json")?
        .call_method1("dumps", (boundary,))?
        .extract()?;
    let boundary_input: Boundary2DInput = serde_json::from_str(&boundary_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid boundary: {}", e)))?;

    // Parse config
    let config_input: Option<ConfigInput> = if let Some(cfg) = config {
        let cfg_json: String = py
            .import_bound("json")?
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

    let rust_config = build_config(config_input);

    // Solve
    let nester = Nester2D::new(rust_config);
    let output = match nester.solve(&rust_geometries, &rust_boundary) {
        Ok(result) => SolveOutput {
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
            unplaced: result.unplaced,
            computation_time_ms: result.computation_time_ms,
            error: None,
        },
        Err(e) => SolveOutput {
            success: false,
            placements: vec![],
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: vec![],
            computation_time_ms: 0,
            error: Some(e.to_string()),
        },
    };

    // Convert to Python dict
    let output_json = serde_json::to_string(&output)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
    let json_module = py.import_bound("json")?;
    let result = json_module.call_method1("loads", (output_json,))?;
    Ok(result.into())
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
fn solve_3d(
    py: Python<'_>,
    geometries: &Bound<'_, PyAny>,
    boundary: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    // Parse geometries
    let geom_json: String = py
        .import_bound("json")?
        .call_method1("dumps", (geometries,))?
        .extract()?;
    let geom_inputs: Vec<Geometry3DInput> = serde_json::from_str(&geom_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid geometries: {}", e)))?;

    // Parse boundary
    let boundary_json: String = py
        .import_bound("json")?
        .call_method1("dumps", (boundary,))?
        .extract()?;
    let boundary_input: Boundary3DInput = serde_json::from_str(&boundary_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid boundary: {}", e)))?;

    // Parse config
    let config_input: Option<ConfigInput> = if let Some(cfg) = config {
        let cfg_json: String = py
            .import_bound("json")?
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

    let rust_config = build_config(config_input);

    // Solve
    let packer = Packer3D::new(rust_config);
    let output = match packer.solve(&rust_geometries, &rust_boundary) {
        Ok(result) => SolveOutput {
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
            unplaced: result.unplaced,
            computation_time_ms: result.computation_time_ms,
            error: None,
        },
        Err(e) => SolveOutput {
            success: false,
            placements: vec![],
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: vec![],
            computation_time_ms: 0,
            error: Some(e.to_string()),
        },
    };

    // Convert to Python dict
    let output_json = serde_json::to_string(&output)
        .map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))?;
    let json_module = py.import_bound("json")?;
    let result = json_module.call_method1("loads", (output_json,))?;
    Ok(result.into())
}

/// Get the library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// List available strategies.
#[pyfunction]
fn available_strategies() -> Vec<&'static str> {
    vec!["blf", "nfp", "ga", "brkga", "sa", "ep"]
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
