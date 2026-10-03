//! Result types for cutting path optimization.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::contour::{ContourId, ContourType};

/// Result of cutting path optimization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CuttingPathResult {
    /// Ordered sequence of cutting steps.
    pub sequence: Vec<CutStep>,

    /// Total cutting distance (sum of all contour perimeters).
    pub total_cut_distance: f64,

    /// Total non-cutting (rapid traverse) distance.
    pub total_rapid_distance: f64,

    /// Total number of piercing operations.
    pub total_pierces: usize,

    /// Computation time in milliseconds.
    pub computation_time_ms: u64,

    /// Estimated total cutting time in seconds (if speeds are configured).
    pub estimated_time_seconds: Option<f64>,

    /// Contours of placed parts that are not in `sequence`, and why. A part
    /// with a skipped contour is not fully cut by this plan.
    pub skipped: Vec<SkippedContour>,
}

/// A contour the plan cannot cut.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SkippedContour {
    /// ID of the contour.
    pub contour_id: ContourId,
    /// ID of the source geometry it belongs to.
    pub geometry_id: String,
    /// Instance index of the placed geometry (0-based).
    pub instance: usize,
    /// Whether it is an exterior or interior contour.
    pub contour_type: ContourType,
    /// Why it was left out.
    pub reason: SkipReason,
}

/// Why a contour is left out of the cutting sequence.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SkipReason {
    /// Offsetting by half the kerf width left no polygon: the contour is
    /// smaller than the tool, so no path keeps the part's dimensions.
    KerfCollapsed {
        /// The offset applied (`+kerf/2` exterior, `-kerf/2` interior).
        offset: f64,
    },
}

impl SkipReason {
    /// Stable snake_case name of the reason, as the JSON API reports it.
    pub fn code(&self) -> &'static str {
        match self {
            SkipReason::KerfCollapsed { .. } => "kerf_collapsed",
        }
    }
}

impl CuttingPathResult {
    /// Creates a new empty result.
    pub fn new() -> Self {
        Self {
            sequence: Vec::new(),
            total_cut_distance: 0.0,
            total_rapid_distance: 0.0,
            total_pierces: 0,
            computation_time_ms: 0,
            estimated_time_seconds: None,
            skipped: Vec::new(),
        }
    }

    /// Returns the total distance (cutting + rapid).
    pub fn total_distance(&self) -> f64 {
        self.total_cut_distance + self.total_rapid_distance
    }

    /// Returns the cutting efficiency (cut distance / total distance).
    pub fn efficiency(&self) -> f64 {
        let total = self.total_distance();
        if total > 0.0 {
            self.total_cut_distance / total
        } else {
            0.0
        }
    }
}

impl CuttingPathResult {
    /// The JSON API response for this result -- one conversion shared by every
    /// binding, so a field cannot reach one transport and not another.
    pub fn to_response(&self) -> u_nesting_core::api_types::CuttingResponse {
        use u_nesting_core::api_types::{CutStepResponse, CuttingResponse, SkippedContourResponse};
        CuttingResponse {
            version: u_nesting_core::api_types::API_VERSION.to_string(),
            success: true,
            error: None,
            sequence: self
                .sequence
                .iter()
                .map(|step| CutStepResponse {
                    contour_id: step.contour_id,
                    geometry_id: step.geometry_id.clone(),
                    instance: step.instance,
                    contour_type: step.contour_type.as_str().to_string(),
                    pierce_point: [step.pierce_point.0, step.pierce_point.1],
                    cut_direction: step.cut_direction.as_str().to_string(),
                    rapid_from: step.rapid_from.map(|p| [p.0, p.1]),
                    rapid_distance: step.rapid_distance,
                    cut_distance: step.cut_distance,
                })
                .collect(),
            total_cut_distance: self.total_cut_distance,
            total_rapid_distance: self.total_rapid_distance,
            total_pierces: self.total_pierces,
            estimated_time_seconds: self.estimated_time_seconds,
            efficiency: self.efficiency(),
            computation_time_ms: self.computation_time_ms,
            skipped_contours: self
                .skipped
                .iter()
                .map(|s| SkippedContourResponse {
                    contour_id: s.contour_id,
                    geometry_id: s.geometry_id.clone(),
                    instance: s.instance,
                    contour_type: s.contour_type.as_str().to_string(),
                    reason: s.reason.code().to_string(),
                    offset: match s.reason {
                        SkipReason::KerfCollapsed { offset } => Some(offset),
                    },
                })
                .collect(),
        }
    }
}

impl Default for CuttingPathResult {
    fn default() -> Self {
        Self::new()
    }
}

/// A single step in the cutting sequence.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CutStep {
    /// ID of the contour being cut.
    pub contour_id: ContourId,

    /// ID of the source geometry this contour belongs to.
    pub geometry_id: String,

    /// Instance index of the placed geometry (0-based).
    pub instance: usize,

    /// Whether this is an exterior or interior contour.
    pub contour_type: ContourType,

    /// Piercing point (entry point on the contour).
    pub pierce_point: (f64, f64),

    /// Cutting direction around the contour.
    pub cut_direction: CutDirection,

    /// Starting point of the rapid move to reach this contour.
    /// None for the first step (starts from home position).
    pub rapid_from: Option<(f64, f64)>,

    /// Distance of the rapid move to reach the pierce point.
    pub rapid_distance: f64,

    /// Perimeter (cutting distance) of this contour.
    pub cut_distance: f64,
}

/// Cutting direction around a contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CutDirection {
    /// Counter-clockwise (conventional for exterior contours).
    Ccw,
    /// Clockwise (conventional for interior/hole contours).
    Cw,
}

impl CutDirection {
    /// The JSON API name: `"ccw"` or `"cw"`.
    pub fn as_str(self) -> &'static str {
        match self {
            CutDirection::Ccw => "ccw",
            CutDirection::Cw => "cw",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_result() {
        let result = CuttingPathResult::new();
        assert!(result.sequence.is_empty());
        assert_eq!(result.total_pierces, 0);
        assert_eq!(result.total_distance(), 0.0);
        assert_eq!(result.efficiency(), 0.0);
    }

    #[test]
    fn test_efficiency() {
        let mut result = CuttingPathResult::new();
        result.total_cut_distance = 800.0;
        result.total_rapid_distance = 200.0;
        assert!((result.efficiency() - 0.8).abs() < 1e-10);
    }
}
