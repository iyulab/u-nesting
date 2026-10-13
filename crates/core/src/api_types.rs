//! Shared API request/response types for FFI and WASM bindings.
//!
//! These types define the JSON schema for U-Nesting's external API.
//! Both the C FFI and WebAssembly bindings share these types to avoid drift.

use serde::{Deserialize, Serialize};

use crate::error::whole_at_least;
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

    /// Quantity to place: a whole number, at least 1.
    #[serde(default = "default_quantity", deserialize_with = "whole::required")]
    pub quantity: i128,

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

    /// Quantity to place: a whole number, at least 1.
    #[serde(default = "default_quantity", deserialize_with = "whole::required")]
    pub quantity: i128,

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

    /// Time limit in milliseconds: a whole number, at least 0.
    #[serde(default, deserialize_with = "whole::optional")]
    pub time_limit_ms: Option<i128>,

    /// Target utilization (0.0 - 1.0).
    pub target_utilization: Option<f64>,

    /// GA/BRKGA population size: a whole number, at least 2. Omitted, each
    /// strategy uses the size it is tuned for (see `Config::population_size`).
    #[serde(default, deserialize_with = "whole::optional")]
    pub population_size: Option<i128>,

    /// GA/BRKGA generation budget: a whole number from 0 to 4294967295.
    /// Omitted, each strategy uses the budget it is tuned for (see
    /// `Config::max_generations`).
    #[serde(default, deserialize_with = "whole::optional")]
    pub max_generations: Option<i128>,

    /// GA crossover rate.
    pub crossover_rate: Option<f64>,

    /// GA mutation rate.
    pub mutation_rate: Option<f64>,

    /// Optional RNG seed for reproducible stochastic runs (GA, BRKGA, SA,
    /// GDRR, ALNS): a whole number from 0 to 18446744073709551615. A seeded
    /// run that ends on its own budget, not on the time limit, is reproduced
    /// exactly (see `Config::seed`).
    #[serde(default, deserialize_with = "whole::optional")]
    pub seed: Option<i128>,

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

fn default_quantity() -> i128 {
    1
}

/// Reads a request's whole numbers -- counts, time budgets, seeds -- as given.
///
/// A count is read as a number, not into an unsigned type: `-1` is a value
/// out of range (refused where the field is judged, as
/// `parameter_out_of_range`), not a request of the wrong shape. A fraction is
/// the wrong shape -- refused here, so as `malformed_input` naming the field.
/// `i128` holds every `u64` (a seed) and its negatives.
mod whole {
    use serde::de::{self, Deserializer, Visitor};
    use std::fmt;

    struct Whole;

    impl<'de> Visitor<'de> for Whole {
        type Value = i128;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a whole number")
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<i128, E> {
            Ok(v.into())
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<i128, E> {
            Ok(v.into())
        }

        fn visit_i128<E: de::Error>(self, v: i128) -> Result<i128, E> {
            Ok(v)
        }

        fn visit_u128<E: de::Error>(self, v: u128) -> Result<i128, E> {
            // Beyond i128 is beyond every range: read it as the largest.
            Ok(i128::try_from(v).unwrap_or(i128::MAX))
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<i128, E> {
            // `3.0` is the whole number 3; a value past i128 saturates, still
            // outside every range.
            if v.is_finite() && v.fract() == 0.0 {
                Ok(v as i128)
            } else {
                Err(E::custom(format_args!("must be a whole number, got {v}")))
            }
        }
    }

    struct OptionalWhole;

    impl<'de> Visitor<'de> for OptionalWhole {
        type Value = Option<i128>;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a whole number or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Option<i128>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<i128>, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Option<i128>, D::Error> {
            required(d).map(Some)
        }
    }

    pub(super) fn required<'de, D: Deserializer<'de>>(d: D) -> Result<i128, D::Error> {
        d.deserialize_any(Whole)
    }

    pub(super) fn optional<'de, D: Deserializer<'de>>(d: D) -> Result<Option<i128>, D::Error> {
        d.deserialize_option(OptionalWhole)
    }
}

/// The request `json` describes, read as given: text that is not JSON, or a
/// value of the wrong shape -- a missing or unknown key, a string for a number,
/// a fraction for a count -- is refused as `malformed_input`, its `parameter`
/// the path of the field at fault (`geometries[2].quantity`) when there is one.
///
/// # Errors
///
/// [`Error::MalformedInput`] as above.
pub fn from_json<T: serde::de::DeserializeOwned>(json: &str) -> crate::Result<T> {
    use serde_json::error::Category;

    // Text that is not JSON names no field: where it broke off is no path.
    let refused = |path: Option<String>, e: serde_json::Error| match e.classify() {
        Category::Syntax | Category::Eof => Error::malformed(None, format!("Invalid JSON: {e}")),
        _ => Error::MalformedInput {
            message: path
                .as_ref()
                .map_or_else(|| e.to_string(), |path| format!("{path}: {e}")),
            parameter: path,
        },
    };
    let mut de = serde_json::Deserializer::from_str(json);
    let value = serde_path_to_error::deserialize(&mut de).map_err(|e| {
        let path = e.path();
        let named = path
            .iter()
            .any(|s| !matches!(s, serde_path_to_error::Segment::Unknown));
        let path = named.then(|| path.to_string());
        refused(path, e.into_inner())
    })?;
    de.end().map_err(|e| refused(None, e))?;
    Ok(value)
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

    /// Z position, along the height axis (the third of `dimensions`).
    pub z: f64,

    /// Orientation label as an axis permutation, e.g. `"xyz"`, `"xzy"`.
    pub orientation: String,
}

impl Pack3DResponse {
    /// The response for a refused request: the message, its `code` and the
    /// values behind it, every setting named by its path in the request
    /// ([`ConfigRequest::locate`]).
    pub fn refused(error: &Error) -> Self {
        let error = &ConfigRequest::locate(error);
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

    /// Maximum number of 2-opt improvement iterations: a whole number, at
    /// least 0.
    #[serde(default, deserialize_with = "whole::optional")]
    pub max_2opt_iterations: Option<i128>,

    /// Wall-clock budget (ms) for the 2-opt improvement phase. `0` = unlimited.
    /// Prevents the sequencing pass from blocking the caller (e.g. the browser
    /// main thread) on large inputs. Defaults to 5000 when omitted.
    #[serde(default, deserialize_with = "whole::optional")]
    pub time_limit_ms: Option<i128>,

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

    /// Number of candidate pierce points per contour: a whole number, at
    /// least 1.
    #[serde(default, deserialize_with = "whole::optional")]
    pub pierce_candidates: Option<i128>,

    /// Tolerance for geometric comparisons.
    pub tolerance: Option<f64>,
}

impl CuttingConfigRequest {
    /// The keys a request's `cutting_config` takes. A refusal of one of
    /// these settings is named, on the wire, by its path there:
    /// `cutting_config.kerf_width`.
    pub const KEYS: [&'static str; 11] = [
        "kerf_width",
        "pierce_weight",
        "max_2opt_iterations",
        "time_limit_ms",
        "rapid_speed",
        "cut_speed",
        "exterior_direction",
        "interior_direction",
        "home_position",
        "pierce_candidates",
        "tolerance",
    ];

    /// `error` as a cutting response reports it: a setting of the request's
    /// `cutting_config` named by its path (`cutting_config.kerf_width`),
    /// whichever step refused it. The Rust API keeps the field name.
    pub fn locate(error: &Error) -> Error {
        error.clone().located("cutting_config", &Self::KEYS)
    }
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

    /// Estimated total time in seconds: the rapid distance at `rapid_speed`
    /// plus the cut distance at `cut_speed` (0 when nothing is cut, and on
    /// error responses).
    pub estimated_time_seconds: f64,

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
    /// values behind it, every setting named by its path in the request
    /// ([`ConfigRequest::locate`]).
    pub fn refused(error: &Error) -> Self {
        let error = &ConfigRequest::locate(error);
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
    /// values behind it, every setting named by its path in the request
    /// ([`CuttingConfigRequest::locate`]).
    pub fn refused(error: &Error) -> Self {
        let error = &CuttingConfigRequest::locate(error);
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
            estimated_time_seconds: 0.0,
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
    /// Closed lower bound of the accepted range: the value may equal it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Closed upper bound of the accepted range: the value may equal it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Open lower bound of the accepted range: the value must be above it.
    /// Never present together with `min`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusive_min: Option<f64>,
    /// Open upper bound of the accepted range: the value must be below it.
    /// Never present together with `max`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusive_max: Option<f64>,
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
                id,
                min,
                max,
                exclusive_min,
                exclusive_max,
                got,
                ..
            } => {
                d.parameter = Some(parameter.clone());
                d.id = id.clone();
                d.min = *min;
                d.max = *max;
                d.exclusive_min = *exclusive_min;
                d.exclusive_max = *exclusive_max;
                d.got = Some(RefusalValue::Number(*got));
            }
            Error::EmptyInput { parameter } | Error::InvalidOption { parameter, .. } => {
                d.parameter = Some(parameter.clone())
            }
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
            // No `got`: JSON has no value for a number that is not finite.
            Error::NotFinite {
                parameter,
                id,
                index,
                ..
            } => {
                d.parameter = Some(parameter.clone());
                d.id = id.clone();
                d.index = *index;
            }
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
    /// The keys a request's `config` takes. A refusal of one of these
    /// settings is named, on the wire, by its path there: `config.spacing`.
    pub const KEYS: [&'static str; 11] = [
        "spacing",
        "margin",
        "strategy",
        "time_limit_ms",
        "target_utilization",
        "population_size",
        "max_generations",
        "crossover_rate",
        "mutation_rate",
        "seed",
        "multi_sheet",
    ];

    /// `error` as a solve response reports it: a setting of the request's
    /// `config` named by its path (`config.spacing`, like `boundary.width` and
    /// `geometries[2].quantity`), whichever step refused it -- reading the
    /// configuration or the solve itself (a strategy the dimension does not
    /// offer). The Rust API keeps the field name (`spacing`).
    pub fn locate(error: &Error) -> Error {
        error.clone().located("config", &Self::KEYS)
    }

    /// The configuration a 3D packing request asks for: as
    /// [`Self::to_config`], with `multi_sheet` -- a 2D setting -- refused
    /// rather than ignored.
    pub fn to_config_3d(&self) -> crate::Result<Config> {
        if self.multi_sheet.is_some() {
            return Err(Error::invalid_option(
                "multi_sheet",
                "multi_sheet spreads 2D parts over sheets; 3D packing has none -- leave it out"
                    .to_string(),
            ));
        }
        self.to_config()
    }

    /// The solver configuration this request asks for, refused as the solver
    /// would refuse it: a strategy name nobody knows is `unknown_option`, a
    /// setting outside its range is `parameter_out_of_range` -- never clamped
    /// into range. A negative count is out of range like a too-small one
    /// (`population_size` -1 and 1 are both refused with `min` 2).
    pub fn to_config(&self) -> crate::Result<Config> {
        let mut config = Config::default();
        if let Some(strategy) = &self.strategy {
            config.strategy = Strategy::parse(strategy).ok_or_else(|| Error::UnknownOption {
                parameter: "strategy".to_string(),
                got: strategy.clone(),
                expected: Strategy::NAMES.iter().map(|n| n.to_string()).collect(),
            })?;
        }
        // A setting only some strategies read is refused when the chosen one
        // does not: run without it, the request would claim a setting that
        // changed nothing. (It used to be accepted and ignored.)
        let present = [
            ("population_size", self.population_size.is_some()),
            ("max_generations", self.max_generations.is_some()),
            ("crossover_rate", self.crossover_rate.is_some()),
            ("mutation_rate", self.mutation_rate.is_some()),
            ("seed", self.seed.is_some()),
            ("target_utilization", self.target_utilization.is_some()),
        ];
        for (setting, given) in present {
            if given && !config.strategy.reads_setting(setting) {
                let name = self.strategy.as_deref().unwrap_or(Strategy::NAMES[0]);
                return Err(Error::invalid_option(
                    setting,
                    format!(
                        "{setting} is read by {} -- strategy \"{name}\" does not read it; leave it out",
                        Strategy::readers_of(setting).join(", ")
                    ),
                ));
            }
        }
        if let Some(spacing) = self.spacing {
            config.spacing = spacing;
        }
        if let Some(margin) = self.margin {
            config.margin = margin;
        }
        if let Some(time_limit) = self.time_limit_ms {
            config.time_limit_ms = whole_at_least("time_limit_ms", time_limit, 0)?;
        }
        if let Some(target) = self.target_utilization {
            config.target_utilization = Some(target);
        }
        if let Some(pop) = self.population_size {
            config.population_size = Some(Config::check_population_size(pop)?);
        }
        if let Some(gens) = self.max_generations {
            config.max_generations = Some(whole_at_least("max_generations", gens, 0)?);
        }
        if let Some(crossover) = self.crossover_rate {
            config.crossover_rate = crossover;
        }
        if let Some(mutation) = self.mutation_rate {
            config.mutation_rate = mutation;
        }
        if let Some(seed) = self.seed {
            config.seed = Some(whole_at_least("seed", seed, 0)?);
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

    /// The keys a refusal is located by are the keys the request types read
    /// -- every one of them, and no other.
    #[test]
    fn the_located_keys_are_the_request_keys() {
        let keys = |v: serde_json::Value| -> Vec<String> {
            let mut k: Vec<String> = v.as_object().expect("an object").keys().cloned().collect();
            k.sort();
            k
        };
        let sorted = |k: &[&str]| -> Vec<String> {
            let mut k: Vec<String> = k.iter().map(|s| s.to_string()).collect();
            k.sort();
            k
        };
        assert_eq!(
            keys(serde_json::to_value(super::ConfigRequest::default()).unwrap()),
            sorted(&super::ConfigRequest::KEYS)
        );
        assert_eq!(
            keys(serde_json::to_value(super::CuttingConfigRequest::default()).unwrap()),
            sorted(&super::CuttingConfigRequest::KEYS)
        );
    }

    /// Every refusal on the wire names a setting by its path in the request,
    /// whichever step refused it; a part already named by its path, and a
    /// request part that is not a setting, keep their names.
    #[test]
    fn wire_refusals_name_settings_by_their_request_path() {
        let parameter = |v: serde_json::Value| v["details"]["parameter"].clone();
        let strategy = crate::Error::invalid_option("strategy", "3D only".into());
        for v in [
            serde_json::to_value(super::SolveResponse::refused(&strategy)).unwrap(),
            serde_json::to_value(super::Pack3DResponse::refused(&strategy)).unwrap(),
        ] {
            assert_eq!(parameter(v.clone()), "config.strategy");
            assert_eq!(v["error"], "config.strategy: 3D only");
        }
        let seed = super::ConfigRequest {
            seed: Some(-1),
            ..Default::default()
        }
        .to_config()
        .expect_err("negative");
        assert_eq!(
            seed.parameter(),
            Some("seed"),
            "the Rust API names the field"
        );
        assert_eq!(
            parameter(serde_json::to_value(super::SolveResponse::refused(&seed)).unwrap()),
            "config.seed"
        );
        let kerf = crate::error::check_at_least("kerf_width", -1.0, 0.0).expect_err("negative");
        assert_eq!(
            parameter(serde_json::to_value(super::CuttingResponse::refused(&kerf)).unwrap()),
            "cutting_config.kerf_width"
        );
        for kept in [
            crate::error::check_above("boundary.width", 0.0, 0.0).expect_err("flat"),
            crate::Error::empty_input("geometries"),
            crate::Error::malformed(Some("config.seed"), "a fraction".into()),
        ] {
            let v = serde_json::to_value(super::SolveResponse::refused(&kept)).unwrap();
            assert_eq!(parameter(v), kept.parameter().expect("named"));
        }
        // Located twice is located once.
        let twice = super::ConfigRequest::locate(&super::ConfigRequest::locate(&seed));
        assert_eq!(twice.parameter(), Some("config.seed"));
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
            id: None,
            min: Some(0.0),
            max: None,
            exclusive_min: None,
            exclusive_max: None,
            got: -1.0,
            range: "a finite number >= 0".into(),
        };
        let v = serde_json::to_value(super::SolveResponse::refused(&range)).expect("serialize");
        assert_eq!(v["success"], false);
        assert_eq!(v["code"], "parameter_out_of_range");
        // The Rust API names the field; the response, the path in the request.
        assert_eq!(
            v["details"],
            json!({ "parameter": "config.spacing", "min": 0.0, "got": -1.0 })
        );
        assert_eq!(
            v["error"],
            "config.spacing must be a finite number >= 0, got -1"
        );

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

    /// A bound the value may not equal is an open one: it goes out as
    /// `exclusive_min` / `exclusive_max`, never as `min` / `max`, and a
    /// closed-only refusal keeps its shape (see the test above).
    #[test]
    fn an_open_bound_is_reported_as_exclusive() {
        let err = crate::sa::SaConfig {
            initial_temp: 0.0,
            ..Default::default()
        }
        .validate()
        .expect_err("a temperature of 0 is refused");
        let v = serde_json::to_value(super::SolveResponse::refused(&err)).expect("serialize");
        assert_eq!(v["code"], "parameter_out_of_range");
        assert_eq!(
            v["details"],
            json!({ "parameter": "initial_temp", "exclusive_min": 0.0, "got": 0.0 })
        );
        assert_eq!(
            v["error"],
            "initial_temp must be a finite number > 0, got 0"
        );

        let grid = crate::exact::ExactConfig {
            grid_step: -1.0,
            ..Default::default()
        }
        .validate()
        .expect_err("a negative grid step is refused");
        assert_eq!(
            serde_json::to_value(super::RefusalDetails::of(&grid)).expect("serialize"),
            json!({ "parameter": "grid_step", "exclusive_min": 0.0, "got": -1.0 })
        );

        // A mixed range -- open below, closed above -- survives the round trip.
        let open = crate::Error::OutOfRange {
            parameter: "ratio".into(),
            id: None,
            min: None,
            max: Some(1.0),
            exclusive_min: Some(0.0),
            exclusive_max: None,
            got: 0.0,
            range: "in (0, 1]".into(),
        };
        let wire = serde_json::to_value(super::SolveResponse::refused(&open)).expect("serialize");
        assert_eq!(
            wire["details"],
            json!({ "parameter": "ratio", "max": 1.0, "exclusive_min": 0.0, "got": 0.0 })
        );
        let back: super::SolveResponse = serde_json::from_value(wire).expect("deserialize");
        assert_eq!(back.details, Some(super::RefusalDetails::of(&open)));
    }

    /// A list with nothing in it names itself and nothing else.
    #[test]
    fn an_empty_list_is_named() {
        let empty = crate::Error::empty_input("geometries");
        let v = serde_json::to_value(super::SolveResponse::refused(&empty)).expect("serialize");
        assert_eq!(v["code"], "empty_input");
        assert_eq!(v["details"], json!({ "parameter": "geometries" }));
        assert_eq!(v["error"], "geometries is empty; give at least one");
    }

    /// The algorithm builders store what they are given -- they used to raise
    /// a value below a floor to the floor (a grid step of 0.05 became 0.1, a
    /// NaN ruin weight 0) -- and `validate` refuses what is out of range.
    #[test]
    fn builders_store_what_they_are_given() {
        use crate::alns::AlnsConfig;
        use crate::brkga::BrkgaConfig;
        use crate::exact::ExactConfig;
        use crate::ga::GaConfig;
        use crate::gdrr::GdrrConfig;
        use crate::sa::SaConfig;
        let details = |r: crate::Result<()>| {
            let e = r.expect_err("out of range");
            (
                e.code(),
                serde_json::to_value(super::RefusalDetails::of(&e)).expect("serialize"),
            )
        };
        let range = |v: serde_json::Value| ("parameter_out_of_range", v);

        let sa = SaConfig::default().with_initial_temp(0.0);
        assert_eq!(sa.initial_temp, 0.0);
        assert_eq!(
            details(sa.validate()),
            range(json!({ "parameter": "initial_temp", "exclusive_min": 0.0, "got": 0.0 }))
        );
        assert_eq!(
            details(SaConfig::default().with_final_temp(0.0).validate()),
            range(json!({ "parameter": "final_temp", "exclusive_min": 0.0, "got": 0.0 }))
        );
        let sa = SaConfig::default().with_iterations_per_temp(0);
        assert_eq!(sa.iterations_per_temp, 0);
        assert_eq!(
            details(sa.validate()),
            range(json!({ "parameter": "iterations_per_temp", "min": 1.0, "got": 0.0 }))
        );
        let sa = SaConfig::default().with_reheating(10, 0.5);
        assert_eq!(sa.reheat_factor, 0.5);
        assert_eq!(
            details(sa.validate()),
            range(json!({ "parameter": "reheat_factor", "min": 1.0, "got": 0.5 }))
        );
        assert!(SaConfig::default()
            .with_reheating(10, 1.05)
            .validate()
            .is_ok());

        let exact = ExactConfig::default().with_grid_step(0.05);
        assert_eq!(exact.grid_step, 0.05);
        assert!(exact.validate().is_ok());
        assert_eq!(
            details(ExactConfig::default().with_grid_step(0.0).validate()),
            range(json!({ "parameter": "grid_step", "exclusive_min": 0.0, "got": 0.0 }))
        );
        assert_eq!(
            details(ExactConfig::default().with_max_items(0).validate()),
            range(json!({ "parameter": "max_items", "min": 1.0, "got": 0.0 }))
        );
        assert_eq!(
            details(ExactConfig::default().with_rotation_steps(0).validate()),
            range(json!({ "parameter": "rotation_steps", "min": 1.0, "got": 0.0 }))
        );

        let gdrr = GdrrConfig::default().with_ruin_weights(-1.0, 1.0, 1.0);
        assert_eq!(gdrr.random_ruin_weight, -1.0);
        assert_eq!(
            details(gdrr.validate()),
            range(json!({ "parameter": "random_ruin_weight", "min": 0.0, "got": -1.0 }))
        );
        let gdrr = GdrrConfig::default().with_ruin_weights(1.0, f64::NAN, 1.0);
        assert!(gdrr.cluster_ruin_weight.is_nan(), "NaN is not 0");
        assert!(gdrr.validate().is_err());

        let alns = AlnsConfig::default().with_segment_size(0);
        assert_eq!(alns.segment_size, 0);
        assert_eq!(
            details(alns.validate()),
            range(json!({ "parameter": "segment_size", "min": 1.0, "got": 0.0 }))
        );

        let ga = GaConfig::default().with_population_size(1);
        assert_eq!(ga.population_size, 1);
        assert_eq!(
            details(ga.validate()),
            range(json!({ "parameter": "population_size", "min": 2.0, "got": 1.0 }))
        );

        // BRKGA used to raise anything below 4 to 4; 3 is a population.
        let brkga = BrkgaConfig::default().with_population_size(3);
        assert_eq!(brkga.population_size, 3);
        assert!(brkga.validate().is_ok());
        assert_eq!(
            details(BrkgaConfig::default().with_population_size(1).validate()),
            range(json!({ "parameter": "population_size", "min": 2.0, "got": 1.0 }))
        );
        // Each fraction rounds up to a whole individual: 2 elite and 2
        // mutants do not fit in 3.
        let crowded = BrkgaConfig::default()
            .with_population_size(3)
            .with_elite_fraction(0.5)
            .with_mutant_fraction(0.5);
        assert_eq!(
            details(crowded.validate()),
            ("invalid_option", json!({ "parameter": "population_size" }))
        );
    }

    #[test]
    fn a_successful_response_has_no_reason() {
        let v = serde_json::to_value(super::SolveResponse::from(crate::SolveResult::<f64>::new()))
            .expect("serialize");
        assert_eq!(v["code"], serde_json::Value::Null);
        assert_eq!(v["details"], serde_json::Value::Null);
    }

    /// A setting only some strategies read is refused under a strategy that
    /// does not read it -- named, with the strategies that do -- rather than
    /// accepted and ignored; under one that reads it, it is taken.
    #[test]
    fn a_setting_the_strategy_does_not_read_is_refused() {
        let config = |v: serde_json::Value| {
            serde_json::from_value::<super::ConfigRequest>(v)
                .expect("a config request")
                .to_config()
        };
        for (body, setting) in [
            (
                json!({ "strategy": "sa", "crossover_rate": 0.9 }),
                "crossover_rate",
            ),
            (
                json!({ "strategy": "brkga", "mutation_rate": 0.1 }),
                "mutation_rate",
            ),
            (
                json!({ "strategy": "sa", "population_size": 40 }),
                "population_size",
            ),
            (
                json!({ "strategy": "nfp", "max_generations": 10 }),
                "max_generations",
            ),
            (json!({ "strategy": "blf", "seed": 7 }), "seed"),
            (
                json!({ "strategy": "gdrr", "target_utilization": 0.8 }),
                "target_utilization",
            ),
            // No strategy: the default (blf) reads none of them.
            (json!({ "population_size": 40 }), "population_size"),
        ] {
            let e = config(body.clone()).unwrap_err();
            assert_eq!(e.code(), "invalid_option", "{body}");
            let details = super::RefusalDetails::of(&e);
            assert_eq!(details.parameter.as_deref(), Some(setting), "{body}");
        }
        let e = config(json!({ "strategy": "sa", "crossover_rate": 0.9 })).unwrap_err();
        assert!(e.to_string().contains("read by ga"), "{e}");

        for body in [
            json!({ "strategy": "ga", "crossover_rate": 0.9, "mutation_rate": 0.1, "population_size": 40, "max_generations": 10, "seed": 7, "target_utilization": 0.8 }),
            json!({ "strategy": "brkga", "population_size": 40, "max_generations": 10, "seed": 7, "target_utilization": 0.8 }),
            json!({ "strategy": "sa", "seed": 7, "target_utilization": 0.8 }),
            json!({ "strategy": "alns", "seed": 7 }),
            json!({ "strategy": "hybridexact", "seed": 7 }),
        ] {
            assert!(config(body.clone()).is_ok(), "{body}");
        }
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
        let target = request(json!({ "strategy": "ga", "target_utilization": 1.5 }))
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

    /// A count is read as a number, so `-1` reaches the range check -- and
    /// meets the same bound as a too-small positive value.
    #[test]
    fn counts_are_whole_numbers_judged_by_their_range() {
        // Under `ga`, which reads every one of these settings: the range is
        // what is being judged, not whether the strategy reads the setting.
        let config = |mut v: serde_json::Value| -> crate::Result<crate::solver::Config> {
            v["strategy"] = json!("ga");
            super::from_json::<super::ConfigRequest>(&v.to_string())?.to_config()
        };
        let details = |e: crate::Error| {
            assert_eq!(e.code(), "parameter_out_of_range", "{e}");
            serde_json::to_value(super::RefusalDetails::of(&e)).expect("serialize")
        };
        for got in [-1, 0, 1] {
            assert_eq!(
                details(config(json!({ "population_size": got })).unwrap_err()),
                json!({ "parameter": "population_size", "min": 2.0, "got": got as f64 })
            );
        }
        for setting in ["time_limit_ms", "max_generations", "seed"] {
            assert_eq!(
                details(config(json!({ setting: -1 })).unwrap_err()),
                json!({ "parameter": setting, "min": 0.0, "got": -1.0 })
            );
        }
        let e = config(json!({ "population_size": -1 })).unwrap_err();
        assert_eq!(
            e.to_string(),
            "population_size must be a whole number >= 2, got -1"
        );

        let ok = config(json!({
            "population_size": 2, "max_generations": 4294967295_u64, "time_limit_ms": 0,
            "seed": u64::MAX, "spacing": null
        }))
        .expect("every value in range");
        assert_eq!(ok.population_size, Some(2));
        assert_eq!(ok.max_generations, Some(u32::MAX));
        let omitted = config(json!({})).expect("defaults");
        assert_eq!(omitted.population_size, None, "the strategy's own");
        assert_eq!(omitted.max_generations, None, "the strategy's own");
        assert_eq!(ok.seed, Some(u64::MAX));
        assert_eq!(config(json!({ "seed": 7.0 })).expect("7").seed, Some(7));
        assert_eq!(config(json!({ "seed": null })).expect("default").seed, None);

        // The configuration's own check is the same one.
        let own = crate::solver::Config {
            population_size: Some(1),
            ..Default::default()
        };
        assert_eq!(
            details(own.validate().unwrap_err()),
            json!({ "parameter": "population_size", "min": 2.0, "got": 1.0 })
        );
    }

    /// A fraction, a string or a value of no number at all is the wrong
    /// shape: `malformed_input`, its `parameter` the path of the field.
    #[test]
    fn a_count_of_the_wrong_shape_is_malformed_with_its_path() {
        let refused = |json: &str| match super::from_json::<super::Request2D>(json) {
            Err(e @ crate::Error::MalformedInput { .. }) => e,
            other => panic!("{other:?}"),
        };
        let request = |quantity: &str| {
            format!(
                r#"{{"geometries": [{{"id": "a", "polygon": [[0,0],[1,0],[1,1]]}},
                    {{"id": "b", "polygon": [[0,0],[1,0],[1,1]], "quantity": {quantity}}}],
                    "boundary": {{"width": 5, "height": 5}}}}"#
            )
        };
        for quantity in ["2.5", "\"2\"", "null", "true"] {
            let e = refused(&request(quantity));
            assert_eq!(
                super::RefusalDetails::of(&e).parameter.as_deref(),
                Some("geometries[1].quantity"),
                "{quantity}: {e}"
            );
        }
        let message = refused(&request("2.5")).to_string();
        assert!(
            message.starts_with("geometries[1].quantity: must be a whole number, got 2.5"),
            "{message}"
        );
        let r: super::Request2D = super::from_json(&request("3")).expect("a whole number");
        assert_eq!(r.geometries[1].quantity, 3);
        assert_eq!(r.geometries[0].quantity, 1, "the default");

        // Not JSON, or JSON with something after it: no field to name.
        for text in ["{", "{} {}"] {
            let e = refused(text);
            assert_eq!(super::RefusalDetails::of(&e).parameter, None, "{e}");
        }

        let cutting =
            super::from_json::<super::CuttingConfigRequest>(r#"{"pierce_candidates": 1.5}"#)
                .unwrap_err();
        assert_eq!(
            super::RefusalDetails::of(&cutting).parameter.as_deref(),
            Some("pierce_candidates")
        );
    }
}
