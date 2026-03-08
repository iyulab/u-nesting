//! Shared API request/response types for FFI and WASM bindings.
//!
//! These types define the JSON schema for U-Nesting's external API.
//! Both the C FFI and WebAssembly bindings share these types to avoid drift.

use serde::{Deserialize, Serialize};

/// API version from Cargo.toml.
pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Request mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// 2D nesting mode.
    #[serde(rename = "2d")]
    D2,
    /// 3D bin packing mode.
    #[serde(rename = "3d")]
    D3,
}

/// Request for 2D nesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request2D {
    /// API version.
    #[serde(default)]
    pub version: Option<String>,

    /// Geometries to place.
    pub geometries: Vec<Geometry2DRequest>,

    /// Boundary definition.
    pub boundary: Boundary2DRequest,

    /// Configuration.
    #[serde(default)]
    pub config: Option<ConfigRequest>,
}

/// Request for 3D bin packing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request3D {
    /// API version.
    #[serde(default)]
    pub version: Option<String>,

    /// Geometries to place.
    pub geometries: Vec<Geometry3DRequest>,

    /// Boundary definition.
    pub boundary: Boundary3DRequest,

    /// Configuration.
    #[serde(default)]
    pub config: Option<ConfigRequest>,
}

/// 2D geometry request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry2DRequest {
    /// Unique identifier.
    pub id: String,

    /// Polygon vertices as [[x, y], ...].
    pub polygon: Vec<[f64; 2]>,

    /// Interior holes (optional).
    #[serde(default)]
    pub holes: Option<Vec<Vec<[f64; 2]>>>,

    /// Quantity to place.
    #[serde(default = "default_quantity")]
    pub quantity: usize,

    /// Allowed rotation angles in degrees.
    #[serde(default)]
    pub rotations: Option<Vec<f64>>,

    /// Allow flipping.
    #[serde(default)]
    pub allow_flip: bool,
}

/// 2D boundary request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boundary2DRequest {
    /// Width for rectangular boundary.
    pub width: Option<f64>,

    /// Height for rectangular boundary.
    pub height: Option<f64>,

    /// Polygon vertices for arbitrary boundary.
    pub polygon: Option<Vec<[f64; 2]>>,
}

/// 3D geometry request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry3DRequest {
    /// Unique identifier.
    pub id: String,

    /// Dimensions [width, depth, height].
    pub dimensions: [f64; 3],

    /// Quantity to place.
    #[serde(default = "default_quantity")]
    pub quantity: usize,

    /// Mass (optional).
    pub mass: Option<f64>,

    /// Orientation constraint.
    #[serde(default)]
    pub orientation: Option<String>,
}

/// 3D boundary request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boundary3DRequest {
    /// Dimensions [width, depth, height].
    pub dimensions: [f64; 3],

    /// Maximum mass (optional).
    pub max_mass: Option<f64>,

    /// Enable gravity constraints.
    #[serde(default)]
    pub gravity: bool,

    /// Enable stability constraints.
    #[serde(default)]
    pub stability: bool,
}

/// Configuration request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigRequest {
    /// Spacing between geometries.
    pub spacing: Option<f64>,

    /// Margin from boundary edges.
    pub margin: Option<f64>,

    /// Optimization strategy.
    pub strategy: Option<String>,

    /// Time limit in milliseconds.
    pub time_limit_ms: Option<u64>,

    /// Target utilization (0.0 - 1.0).
    pub target_utilization: Option<f64>,

    /// GA population size.
    pub population_size: Option<usize>,

    /// GA max generations.
    pub max_generations: Option<u32>,

    /// GA crossover rate.
    pub crossover_rate: Option<f64>,

    /// GA mutation rate.
    pub mutation_rate: Option<f64>,
}

/// Response for solve operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResponse {
    /// API version.
    pub version: String,

    /// Whether the operation succeeded.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,

    /// Placements.
    #[serde(default)]
    pub placements: Vec<PlacementResponse>,

    /// Number of boundaries used.
    pub boundaries_used: usize,

    /// Utilization ratio.
    pub utilization: f64,

    /// IDs of unplaced geometries.
    #[serde(default)]
    pub unplaced: Vec<String>,

    /// Computation time in milliseconds.
    pub computation_time_ms: u64,
}

/// Placement response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementResponse {
    /// Geometry ID.
    pub geometry_id: String,

    /// Instance index.
    pub instance: usize,

    /// Position [x, y] or [x, y, z].
    pub position: Vec<f64>,

    /// Rotation angle(s).
    pub rotation: Vec<f64>,

    /// Boundary index.
    pub boundary_index: usize,
}

fn default_quantity() -> usize {
    1
}

impl From<crate::Placement<f64>> for PlacementResponse {
    fn from(p: crate::Placement<f64>) -> Self {
        Self {
            geometry_id: p.geometry_id,
            instance: p.instance,
            position: p.position,
            rotation: p.rotation,
            boundary_index: p.boundary_index,
        }
    }
}

impl<S: Into<f64> + Copy> From<crate::SolveResult<S>> for SolveResponse {
    fn from(r: crate::SolveResult<S>) -> Self {
        Self {
            version: API_VERSION.to_string(),
            success: true,
            error: None,
            placements: Vec::new(), // Converted separately due to type constraints
            boundaries_used: r.boundaries_used,
            utilization: r.utilization,
            unplaced: r.unplaced,
            computation_time_ms: r.computation_time_ms,
        }
    }
}

// --- Cutting Path Types ---

/// Request for cutting path optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuttingRequest {
    /// Original geometry definitions (same format as nesting request).
    pub geometries: Vec<Geometry2DRequest>,

    /// Solve result from a previous nesting operation.
    pub solve_result: SolveResponse,

    /// Cutting path configuration (optional; defaults will be used if absent).
    #[serde(default)]
    pub cutting_config: Option<CuttingConfigRequest>,
}

/// Cutting path configuration request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CuttingConfigRequest {
    /// Kerf width (cutting tool width). Set to 0.0 to disable kerf compensation.
    pub kerf_width: Option<f64>,

    /// Weight factor for pierce count in cost function.
    pub pierce_weight: Option<f64>,

    /// Maximum number of 2-opt improvement iterations.
    pub max_2opt_iterations: Option<usize>,

    /// Machine rapid traverse speed (units/s). For time estimation only.
    pub rapid_speed: Option<f64>,

    /// Machine cutting speed (units/s). For time estimation only.
    pub cut_speed: Option<f64>,

    /// Default cut direction for exterior contours: "ccw", "cw", or "auto".
    pub exterior_direction: Option<String>,

    /// Default cut direction for interior contours: "ccw", "cw", or "auto".
    pub interior_direction: Option<String>,

    /// Home position [x, y] for the cutting head.
    pub home_position: Option<[f64; 2]>,

    /// Number of candidate pierce points per contour.
    pub pierce_candidates: Option<usize>,

    /// Tolerance for geometric comparisons.
    pub tolerance: Option<f64>,
}

/// Response for cutting path optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuttingResponse {
    /// API version.
    pub version: String,

    /// Whether the operation succeeded.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,

    /// Ordered sequence of cutting steps.
    #[serde(default)]
    pub sequence: Vec<CutStepResponse>,

    /// Total cutting distance.
    pub total_cut_distance: f64,

    /// Total non-cutting (rapid traverse) distance.
    pub total_rapid_distance: f64,

    /// Total number of pierce operations.
    pub total_pierces: usize,

    /// Estimated total time in seconds (if speeds configured).
    pub estimated_time_seconds: Option<f64>,

    /// Cutting efficiency (cut_distance / total_distance).
    pub efficiency: f64,

    /// Computation time in milliseconds.
    pub computation_time_ms: u64,
}

/// A single step in the cutting sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutStepResponse {
    /// Contour ID.
    pub contour_id: usize,

    /// Source geometry ID.
    pub geometry_id: String,

    /// Instance index of the placed geometry.
    pub instance: usize,

    /// Contour type: "exterior" or "interior".
    pub contour_type: String,

    /// Piercing point [x, y].
    pub pierce_point: [f64; 2],

    /// Cutting direction: "ccw" or "cw".
    pub cut_direction: String,

    /// Starting point of rapid move [x, y] (null for first step).
    pub rapid_from: Option<[f64; 2]>,

    /// Rapid move distance.
    pub rapid_distance: f64,

    /// Cutting distance (contour perimeter).
    pub cut_distance: f64,
}

impl SolveResponse {
    /// Creates an error response.
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(msg.into()),
            placements: Vec::new(),
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: Vec::new(),
            computation_time_ms: 0,
        }
    }
}

impl CuttingResponse {
    /// Creates an error response.
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(msg.into()),
            sequence: Vec::new(),
            total_cut_distance: 0.0,
            total_rapid_distance: 0.0,
            total_pierces: 0,
            estimated_time_seconds: None,
            efficiency: 0.0,
            computation_time_ms: 0,
        }
    }
}
