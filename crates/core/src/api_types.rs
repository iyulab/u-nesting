//! Shared API request/response types for FFI and WASM bindings.
//!
//! These types define the JSON schema for U-Nesting's external API.
//! Both the C FFI and WebAssembly bindings share these types to avoid drift.

use serde::{Deserialize, Serialize};

use crate::solver::{Config, Strategy};
use crate::Error;

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
#[serde(deny_unknown_fields)]
pub struct Request2D {
    /// API version.
    #[serde(default)]
    pub version: Option<String>,

    /// Mode discriminator used by the auto-detect `solve` entry point.
    /// Accepted (and validated) here so a dispatched request round-trips
    /// under the strict unknown-field policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<Mode>,

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
#[serde(deny_unknown_fields)]
pub struct Request3D {
    /// API version.
    #[serde(default)]
    pub version: Option<String>,

    /// Mode discriminator used by the auto-detect `solve` entry point.
    /// Accepted (and validated) here so a dispatched request round-trips
    /// under the strict unknown-field policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<Mode>,

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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

    /// Optional RNG seed for reproducible stochastic runs (GA, BRKGA, SA).
    pub seed: Option<u64>,

    /// Distribute overflow across multiple sheets (2D only).
    ///
    /// When `true`, parts that do not fit on a single sheet spill onto additional
    /// sheets instead of becoming unplaced. `sheets_used` then reports the sheet
    /// count and each placement's `sheet_index` selects its sheet; placement
    /// coordinates are sheet-local (relative to each sheet's origin). Defaults to
    /// `false` (single-sheet solve).
    pub multi_sheet: Option<bool>,
}

/// Response for solve operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveResponse {
    /// API version.
    pub version: String,

    /// Whether the operation succeeded.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,

    /// Stable name of the reason a request was refused
    /// (`parameter_out_of_range`, `invalid_geometry`, `duplicate_id`, ...);
    /// `null` on success. Branch on this, not on `error`, whose wording may
    /// change.
    #[serde(default)]
    pub code: Option<String>,

    /// The values behind `code` -- the setting, the geometry id, the
    /// positions; `null` on success.
    #[serde(default)]
    pub details: Option<RefusalDetails>,

    /// Placements.
    #[serde(default)]
    pub placements: Vec<PlacementResponse>,

    /// Number of sheets/bins used.
    pub sheets_used: usize,

    /// Utilization ratio.
    pub utilization: f64,

    /// Total number of geometry **instances** requested (Σ of every geometry's
    /// quantity). `placements` is instance-level while `unplaced` lists unique
    /// geometry IDs (deduplicated); use `total_requested - placements.len()` to
    /// get the instance-level unplaced count. `0` on error responses.
    #[serde(default)]
    pub total_requested: usize,

    /// IDs of unplaced geometries (deduplicated). Tells *which* geometries have
    /// at least one unplaced instance; see `unplaced_count` for *how many*.
    #[serde(default)]
    pub unplaced: Vec<String>,

    /// Instance-level count of geometry instances that could not be placed
    /// (`total_requested - placements.len()`). Satisfies the invariant
    /// `placements.len() + unplaced_count == total_requested`. Unlike `unplaced`
    /// (deduplicated IDs), this never undercounts a multi-quantity geometry.
    /// `0` on error responses.
    #[serde(default)]
    pub unplaced_count: usize,

    /// Whether every requested instance was placed
    /// (`placements.len() == total_requested`). Prefer this over `success` to
    /// detect partial packing: `success` only means the solve completed without
    /// error, **not** that all pieces fit. `false` on error responses.
    #[serde(default)]
    pub all_placed: bool,

    /// Axis-aligned bounding box `[width, height]` of the placed pieces' actual
    /// footprint. Boundary-padding independent, unlike `utilization` (which
    /// divides by the full boundary and shrinks as boundary height grows). For an
    /// open-ended roll the larger axis is the material length consumed.
    ///
    /// Populated for **single-sheet** solves (the common open-roll / fabric case),
    /// where all placements share one boundary-local frame. In **multi-sheet**
    /// solves the placements are re-localized per sheet, so a single footprint is
    /// ill-defined and this stays `[0.0, 0.0]`. Also `[0.0, 0.0]` when nothing was
    /// placed or on error.
    #[serde(default)]
    pub used_bounding_box: [f64; 2],

    /// Utilization against the used bounding box
    /// (`placed_area / (used_width * used_height)`) rather than the full boundary.
    /// A **packing-density** metric: the denominator shrinks on **both** axes to
    /// the placed footprint, so it answers "how tightly are the pieces packed
    /// within their own extent?" independent of boundary padding.
    ///
    /// This is **not** a fixed-width stock-efficiency metric. For fixed-width
    /// material (fabric rolls, coil, sheet stock) the boundary width is real
    /// consumed stock, not padding — collapsing the width axis over-reports how
    /// much material was saved. Fixed-width consumers should compute
    /// `placed_area / (boundary_width * used_bounding_box[1])` instead
    /// (`used_bounding_box[1]` is the consumed length and is boundary-independent).
    ///
    /// `0.0` when nothing placed, on error, or in multi-sheet solves
    /// (see `used_bounding_box`).
    #[serde(default)]
    pub used_utilization: f64,

    /// Computation time in milliseconds.
    pub elapsed_ms: u64,
}

/// Placement response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementResponse {
    /// Geometry ID.
    #[serde(rename = "id")]
    pub geometry_id: String,

    /// Instance index (0-based) when multiple copies of the same geometry are placed.
    pub instance: usize,

    /// X position.
    pub x: f64,

    /// Y position.
    pub y: f64,

    /// Rotation angle in degrees.
    pub rotation: f64,

    /// Sheet/bin index (0-based).
    pub sheet_index: usize,

    /// Whether the geometry was flipped/mirrored.
    pub flipped: bool,
}

fn default_quantity() -> usize {
    1
}

impl From<crate::Placement<f64>> for PlacementResponse {
    fn from(p: crate::Placement<f64>) -> Self {
        Self {
            geometry_id: p.geometry_id,
            instance: p.instance,
            x: p.position.first().copied().unwrap_or(0.0),
            y: p.position.get(1).copied().unwrap_or(0.0),
            rotation: p.rotation.first().copied().unwrap_or(0.0).to_degrees(),
            sheet_index: p.boundary_index,
            flipped: p.mirrored,
        }
    }
}

impl<S: Into<f64> + Copy> From<crate::SolveResult<S>> for SolveResponse {
    fn from(r: crate::SolveResult<S>) -> Self {
        // Instance-level accounting: `placements` is instance-level, `total_requested`
        // is Σ quantity. `unplaced_count` is derived from these (not from the
        // deduplicated `unplaced` ID list, which undercounts multi-quantity geoms).
        let placed = r.placements.len();
        let unplaced_count = r.total_requested.saturating_sub(placed);
        let [uw, uh] = r.used_bounding_box;
        let used_area = uw * uh;
        let used_utilization = if used_area > 0.0 {
            r.total_piece_area / used_area
        } else {
            0.0
        };
        Self {
            version: API_VERSION.to_string(),
            success: true,
            error: None,
            code: None,
            details: None,
            placements: Vec::new(), // Converted separately due to type constraints
            sheets_used: r.boundaries_used,
            utilization: r.utilization,
            total_requested: r.total_requested,
            unplaced: r.unplaced,
            unplaced_count,
            all_placed: unplaced_count == 0,
            used_bounding_box: r.used_bounding_box,
            used_utilization,
            elapsed_ms: r.computation_time_ms,
        }
    }
}

/// 3D packing response.
///
/// Distinct from [`SolveResponse`] (used for 2D) because the 3D wire contract
/// differs: bins instead of sheets, and placements carry depth (`z`) and an
/// `orientation` label. Mirrors the C# `PackingResult` binding
/// (bindings/csharp/UNesting/Models/Geometry3D.cs).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pack3DResponse {
    /// API version.
    pub version: String,

    /// Whether the operation succeeded.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,

    /// Stable name of the reason a request was refused
    /// (`parameter_out_of_range`, `invalid_geometry`, `duplicate_id`, ...);
    /// `null` on success. Branch on this, not on `error`, whose wording may
    /// change.
    #[serde(default)]
    pub code: Option<String>,

    /// The values behind `code` -- the setting, the geometry id, the
    /// positions; `null` on success.
    #[serde(default)]
    pub details: Option<RefusalDetails>,

    /// Placements.
    #[serde(default)]
    pub placements: Vec<Placement3DResponse>,

    /// Number of bins used.
    pub bins_used: usize,

    /// Volume utilization ratio.
    pub utilization: f64,

    /// Total number of geometry **instances** requested (Σ of every geometry's
    /// quantity). `placements` is instance-level while `unplaced` lists unique
    /// geometry IDs (deduplicated); use `total_requested - placements.len()` to
    /// get the instance-level unplaced count. `0` on error responses.
    #[serde(default)]
    pub total_requested: usize,

    /// IDs of unplaced geometries (deduplicated). See `unplaced_count` for the
    /// instance-level count.
    #[serde(default)]
    pub unplaced: Vec<String>,

    /// Instance-level count of geometry instances that could not be placed
    /// (`total_requested - placements.len()`). Satisfies the invariant
    /// `placements.len() + unplaced_count == total_requested`. `0` on error.
    #[serde(default)]
    pub unplaced_count: usize,

    /// Whether every requested instance was placed. Prefer this over `success`
    /// to detect partial packing. `false` on error responses.
    #[serde(default)]
    pub all_placed: bool,

    /// Computation time in milliseconds.
    pub elapsed_ms: u64,
}

/// 3D placement response.
///
/// Mirrors the C# `Placement3D` binding: carries `z` and a string
/// `orientation` label (e.g. `"xyz"`), and uses `bin_index` (not `sheet_index`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement3DResponse {
    /// Geometry ID.
    #[serde(rename = "id")]
    pub geometry_id: String,

    /// Instance index (0-based) when multiple copies of the same geometry are placed.
    pub instance: usize,

    /// Bin index (0-based).
    pub bin_index: usize,

    /// X position.
    pub x: f64,

    /// Y position.
    pub y: f64,

    /// Z position (depth).
    pub z: f64,

    /// Orientation label as an axis permutation, e.g. `"xyz"`, `"xzy"`.
    pub orientation: String,
}

impl Pack3DResponse {
    /// The response for a refused request: the message, its `code` and the
    /// values behind it.
    pub fn refused(error: &Error) -> Self {
        Self {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(error.to_string()),
            code: Some(error.code().to_string()),
            details: Some(RefusalDetails::of(error)),
            placements: Vec::new(),
            bins_used: 0,
            utilization: 0.0,
            total_requested: 0,
            unplaced: Vec::new(),
            unplaced_count: 0,
            all_placed: false,
            elapsed_ms: 0,
        }
    }
}

// --- Cutting Path Types ---

/// Request for cutting path optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct CuttingConfigRequest {
    /// Kerf width (cutting tool width). Set to 0.0 to disable kerf compensation.
    pub kerf_width: Option<f64>,

    /// Weight factor for pierce count in cost function.
    pub pierce_weight: Option<f64>,

    /// Maximum number of 2-opt improvement iterations.
    pub max_2opt_iterations: Option<usize>,

    /// Wall-clock budget (ms) for the 2-opt improvement phase. `0` = unlimited.
    /// Prevents the sequencing pass from blocking the caller (e.g. the browser
    /// main thread) on large inputs. Defaults to 5000 when omitted.
    pub time_limit_ms: Option<u64>,

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
#[serde(deny_unknown_fields)]
pub struct CuttingResponse {
    /// API version.
    pub version: String,

    /// Whether the operation succeeded.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,

    /// Stable name of the reason a request was refused
    /// (`parameter_out_of_range`, `invalid_geometry`, `duplicate_id`, ...);
    /// `null` on success. Branch on this, not on `error`, whose wording may
    /// change.
    #[serde(default)]
    pub code: Option<String>,

    /// The values behind `code` -- the setting, the geometry id, the
    /// positions; `null` on success.
    #[serde(default)]
    pub details: Option<RefusalDetails>,

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

    /// Contours of placed parts that the plan cannot cut, and why. A part
    /// listed here is not fully cut by `sequence`.
    #[serde(default)]
    pub skipped_contours: Vec<SkippedContourResponse>,
}

/// A contour left out of the cutting sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkippedContourResponse {
    /// Contour ID.
    pub contour_id: usize,

    /// Source geometry ID.
    pub geometry_id: String,

    /// Instance index of the placed geometry.
    pub instance: usize,

    /// Contour type: "exterior" or "interior".
    pub contour_type: String,

    /// Why it was left out: `"kerf_collapsed"` -- offsetting by half the kerf
    /// width left nothing (the contour is smaller than the tool).
    pub reason: String,

    /// The offset applied when `reason` is `"kerf_collapsed"`.
    pub offset: Option<f64>,
}

/// A single step in the cutting sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// The response for a refused request: the message, its `code` and the
    /// values behind it.
    pub fn refused(error: &Error) -> Self {
        Self {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(error.to_string()),
            code: Some(error.code().to_string()),
            details: Some(RefusalDetails::of(error)),
            placements: Vec::new(),
            sheets_used: 0,
            utilization: 0.0,
            total_requested: 0,
            unplaced: Vec::new(),
            unplaced_count: 0,
            all_placed: false,
            used_bounding_box: [0.0, 0.0],
            used_utilization: 0.0,
            elapsed_ms: 0,
        }
    }
}

impl CuttingResponse {
    /// The response for a refused request: the message, its `code` and the
    /// values behind it.
    pub fn refused(error: &Error) -> Self {
        Self {
            version: API_VERSION.to_string(),
            success: false,
            error: Some(error.to_string()),
            code: Some(error.code().to_string()),
            details: Some(RefusalDetails::of(error)),
            sequence: Vec::new(),
            total_cut_distance: 0.0,
            total_rapid_distance: 0.0,
            total_pierces: 0,
            estimated_time_seconds: None,
            efficiency: 0.0,
            computation_time_ms: 0,
            skipped_contours: Vec::new(),
        }
    }
}

/// The values behind a refusal's `code`. Only the ones the reason has are
/// present.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusalDetails {
    /// The setting or request part at fault (`spacing`, `strategy`,
    /// `geometries`, `max_mass`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    /// The geometry the refusal is about.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// For `duplicate_id`, the position where the id appeared first; `index`
    /// is where it appeared again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first: Option<usize>,
    /// The position of the offending entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Lower bound of the accepted range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Upper bound of the accepted range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// The value given: a number for a range, a name for an unknown option.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub got: Option<RefusalValue>,
    /// The names an unknown option could have been.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Vec<String>>,
}

/// A refused value as it was given.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RefusalValue {
    /// A number outside its range.
    Number(f64),
    /// A name nobody knows.
    Name(String),
}

impl RefusalDetails {
    /// The fields of `error`'s reason.
    pub fn of(error: &Error) -> Self {
        let mut d = RefusalDetails::default();
        match error {
            Error::OutOfRange {
                parameter,
                min,
                max,
                got,
                ..
            } => {
                d.parameter = Some(parameter.clone());
                d.min = *min;
                d.max = *max;
                d.got = Some(RefusalValue::Number(*got));
            }
            Error::InvalidOption { parameter, .. } => d.parameter = Some(parameter.clone()),
            Error::UnknownOption {
                parameter,
                got,
                expected,
            } => {
                d.parameter = Some(parameter.clone());
                d.got = Some(RefusalValue::Name(got.clone()));
                d.expected = Some(expected.clone());
            }
            Error::MalformedInput { parameter, .. } => d.parameter = parameter.clone(),
            Error::InvalidGeometry { id, .. } => d.id = id.clone(),
            Error::DuplicateId { id, first, second } => {
                d.parameter = Some("geometries".to_string());
                d.id = Some(id.clone());
                d.first = Some(*first);
                d.index = Some(*second);
            }
            Error::InvalidBoundary { parameter, .. } => d.parameter = parameter.clone(),
            _ => {}
        }
        d
    }
}

impl ConfigRequest {
    /// The solver configuration this request asks for, refused as the solver
    /// would refuse it: a strategy name nobody knows is `unknown_option`, a
    /// setting outside its range is `parameter_out_of_range` -- never clamped
    /// into range.
    pub fn to_config(&self) -> crate::Result<Config> {
        let mut config = Config::default();
        if let Some(strategy) = &self.strategy {
            config.strategy = Strategy::parse(strategy).ok_or_else(|| Error::UnknownOption {
                parameter: "strategy".to_string(),
                got: strategy.clone(),
                expected: Strategy::NAMES.iter().map(|n| n.to_string()).collect(),
            })?;
        }
        if let Some(spacing) = self.spacing {
            config.spacing = spacing;
        }
        if let Some(margin) = self.margin {
            config.margin = margin;
        }
        if let Some(time_limit) = self.time_limit_ms {
            config.time_limit_ms = time_limit;
        }
        if let Some(target) = self.target_utilization {
            config.target_utilization = Some(target);
        }
        if let Some(pop) = self.population_size {
            config.population_size = pop;
        }
        if let Some(gens) = self.max_generations {
            config.max_generations = gens;
        }
        if let Some(crossover) = self.crossover_rate {
            config.crossover_rate = crossover;
        }
        if let Some(mutation) = self.mutation_rate {
            config.mutation_rate = mutation;
        }
        if let Some(seed) = self.seed {
            config.seed = Some(seed);
        }
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn request_2d_rejects_unknown_keys() {
        assert_rejects_unknown::<super::Request2D>(json!({
            "geometries": [{ "id": "g1", "polygon": [[0.0,0.0],[1.0,0.0],[1.0,1.0]] }],
            "boundary": { "width": 10.0, "height": 10.0 },
            "sheets": 2
        }));
    }

    #[test]
    fn request_3d_nested_rejects_unknown_keys() {
        assert_rejects_unknown::<super::Geometry3DRequest>(json!({
            "id": "b1", "dimensions": [1.0, 1.0, 1.0], "weight": 5.0
        }));
        assert_rejects_unknown::<super::Boundary3DRequest>(json!({
            "dimensions": [10.0, 10.0, 10.0], "max_weight": 100.0
        }));
    }

    #[test]
    fn config_rejects_unknown_keys() {
        assert_rejects_unknown::<super::ConfigRequest>(json!({
            "spacing": 1.0, "rotation_step": 90
        }));
        assert_rejects_unknown::<super::CuttingConfigRequest>(json!({
            "kerf_width": 0.2, "kerf": 0.2
        }));
    }

    #[test]
    fn solve_response_total_requested_defaults_when_absent() {
        // Payloads produced before `total_requested` existed (and the cutting
        // passthrough's inline solve_result) must still deserialize — the field
        // is `#[serde(default)]`, so a missing key yields 0 rather than an error.
        let r: super::SolveResponse = serde_json::from_value(json!({
            "version": "1", "success": true, "placements": [],
            "sheets_used": 0, "utilization": 0.0, "unplaced": [], "elapsed_ms": 0
        }))
        .expect("legacy payload without total_requested must deserialize");
        assert_eq!(r.total_requested, 0);
    }

    #[test]
    fn solve_response_serializes_total_requested() {
        let resp = super::SolveResponse::refused(&crate::Error::Internal("x".into()));
        let v = serde_json::to_value(&resp).expect("serialize");
        assert!(
            v.get("total_requested").is_some(),
            "total_requested must be present in the wire output"
        );
    }

    /// A refusal carries its reason as `code` and the values behind it in
    /// `details`, so a caller can point at the input without reading `error`.
    #[test]
    fn a_refusal_names_its_reason_and_values() {
        let range = crate::Error::OutOfRange {
            parameter: "spacing".into(),
            min: Some(0.0),
            max: None,
            got: -1.0,
            range: "a finite number >= 0".into(),
        };
        let v = serde_json::to_value(super::SolveResponse::refused(&range)).expect("serialize");
        assert_eq!(v["success"], false);
        assert_eq!(v["code"], "parameter_out_of_range");
        assert_eq!(
            v["details"],
            json!({ "parameter": "spacing", "min": 0.0, "got": -1.0 })
        );
        assert_eq!(v["error"], "spacing must be a finite number >= 0, got -1");

        let twice = crate::Error::DuplicateId {
            id: "plate".into(),
            first: 0,
            second: 2,
        };
        let v = serde_json::to_value(super::Pack3DResponse::refused(&twice)).expect("serialize");
        assert_eq!(v["code"], "duplicate_id");
        assert_eq!(
            v["details"],
            json!({ "parameter": "geometries", "id": "plate", "first": 0, "index": 2 })
        );

        // A refused response read back -- as a cutting request carries one --
        // keeps its reason.
        let back: super::SolveResponse = serde_json::from_value(
            serde_json::to_value(super::SolveResponse::refused(&range)).unwrap(),
        )
        .expect("deserialize");
        assert_eq!(back.code.as_deref(), Some("parameter_out_of_range"));
    }

    #[test]
    fn a_successful_response_has_no_reason() {
        let v = serde_json::to_value(super::SolveResponse::from(crate::SolveResult::<f64>::new()))
            .expect("serialize");
        assert_eq!(v["code"], serde_json::Value::Null);
        assert_eq!(v["details"], serde_json::Value::Null);
    }

    #[test]
    fn config_requests_are_refused_not_clamped() {
        let request = |v: serde_json::Value| -> super::ConfigRequest {
            serde_json::from_value(v).expect("a config request")
        };
        let unknown = request(json!({ "strategy": "tabu" }))
            .to_config()
            .unwrap_err();
        assert_eq!(unknown.code(), "unknown_option");
        let details = super::RefusalDetails::of(&unknown);
        assert_eq!(details.parameter.as_deref(), Some("strategy"));
        assert_eq!(details.got, Some(super::RefusalValue::Name("tabu".into())));
        assert!(details.expected.unwrap().contains(&"brkga".to_string()));

        // It was clamped into [0, 1] by every binding.
        let target = request(json!({ "target_utilization": 1.5 }))
            .to_config()
            .unwrap_err();
        assert_eq!(target.code(), "parameter_out_of_range");
        assert_eq!(
            super::RefusalDetails::of(&target).parameter.as_deref(),
            Some("target_utilization")
        );

        let spacing = request(json!({ "spacing": -1.0 })).to_config().unwrap_err();
        assert_eq!(
            super::RefusalDetails::of(&spacing).parameter.as_deref(),
            Some("spacing")
        );

        let ok = request(json!({ "strategy": "ga", "spacing": 2.0, "target_utilization": 0.9 }))
            .to_config()
            .expect("a valid config");
        assert_eq!(ok.spacing, 2.0);
    }

    #[test]
    fn pack3d_response_total_requested_defaults_when_absent() {
        let r: super::Pack3DResponse = serde_json::from_value(json!({
            "version": "1", "success": true, "placements": [],
            "bins_used": 0, "utilization": 0.0, "unplaced": [], "elapsed_ms": 0
        }))
        .expect("legacy 3D payload without total_requested must deserialize");
        assert_eq!(r.total_requested, 0);
    }

    #[test]
    fn solve_response_accounting_invariant_holds() {
        // A result with 6 placed instances out of 20 requested (14 unplaced),
        // with `unplaced` deduplicated to a single geometry ID — the exact
        // silent-loss shape the reporter observed.
        let mut r: crate::SolveResult<f64> = crate::SolveResult::new();
        for i in 0..6 {
            r.placements
                .push(crate::Placement::new_2d("p".to_string(), i, 0.0, 0.0, 0.0));
        }
        r.unplaced.push("p".to_string()); // deduplicated single ID
        r.total_requested = 20;
        r.total_piece_area = 60.0;
        r.used_bounding_box = [10.0, 12.0];

        // The `From` impl derives accounting from the result's placement count
        // (6), then leaves `placements` for the binding to fill. The invariant
        // `placed + unplaced_count == total` therefore holds against the source
        // placement count, not `resp.placements` (empty at this layer).
        let placed = 6;
        let resp = super::SolveResponse::from(r);
        assert_eq!(resp.unplaced_count, 14, "instance-level unplaced count");
        assert_eq!(
            placed + resp.unplaced_count,
            resp.total_requested,
            "invariant placed + unplaced_count == total_requested"
        );
        assert!(!resp.all_placed, "not all placed");
        assert_eq!(resp.used_bounding_box, [10.0, 12.0]);
        // used_utilization = piece_area / (used_w * used_h) = 60 / 120 = 0.5
        assert!((resp.used_utilization - 0.5).abs() < 1e-9);
    }

    #[test]
    fn solve_response_all_placed_when_complete() {
        let mut r: crate::SolveResult<f64> = crate::SolveResult::new();
        for i in 0..3 {
            r.placements
                .push(crate::Placement::new_2d("p".to_string(), i, 0.0, 0.0, 0.0));
        }
        r.total_requested = 3;
        let resp = super::SolveResponse::from(r);
        assert!(resp.all_placed);
        assert_eq!(resp.unplaced_count, 0);
    }

    #[test]
    fn cutting_request_rejects_unknown_keys() {
        assert_rejects_unknown::<super::CuttingRequest>(json!({
            "geometries": [],
            "solve_result": {
                "version": "1", "placements": [], "sheets_used": 0,
                "utilization": 0.0, "elapsed_ms": 0.0, "unplaced": []
            },
            "config": {}
        }));
    }
}
