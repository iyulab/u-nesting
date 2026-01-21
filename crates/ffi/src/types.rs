//! FFI type definitions.

use serde::{Deserialize, Serialize};

/// API version.
pub const API_VERSION: &str = "1.0";

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

impl From<u_nesting_core::Placement<f64>> for PlacementResponse {
    fn from(p: u_nesting_core::Placement<f64>) -> Self {
        Self {
            geometry_id: p.geometry_id,
            instance: p.instance,
            position: p.position,
            rotation: p.rotation,
            boundary_index: p.boundary_index,
        }
    }
}

impl<S: Into<f64> + Copy> From<u_nesting_core::SolveResult<S>> for SolveResponse {
    fn from(r: u_nesting_core::SolveResult<S>) -> Self {
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
