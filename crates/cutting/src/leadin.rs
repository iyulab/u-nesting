//! Lead-in and lead-out path generation.
//!
//! Generates entry and exit paths at pierce points to ensure smooth tool
//! engagement and disengagement with the cut contour.
//!
//! # Lead-In Types
//!
//! - **Line**: Straight approach at a configurable angle to the contour edge
//! - **Arc**: Circular arc approach tangent to the contour at the pierce point
//!
//! # Placement Rules
//!
//! - Exterior contours: lead-in positioned **outside** the part boundary
//! - Interior contours (holes): lead-in positioned **inside** the hole
//!   (i.e., in the waste material that will be removed)
//!
//! # References
//!
//! - Industry standard: lead-in length ~2x kerf width, 45-degree angle

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::contour::{ContourType, CutContour};
use crate::pierce::PierceSelection;
use crate::result::CutDirection;

/// Configuration for lead-in/lead-out generation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LeadInConfig {
    /// Type of lead-in path.
    pub lead_in_type: LeadInType,
    /// Lead-in length (distance from start of lead-in to pierce point).
    pub lead_in_length: f64,
    /// Lead-out length (distance from contour close point to end of lead-out).
    pub lead_out_length: f64,
    /// Angle in radians for line lead-in (relative to the contour tangent at pierce).
    /// Default: PI/4 (45 degrees).
    pub lead_in_angle: f64,
    /// Number of arc segments for arc-type lead-in.
    pub arc_segments: usize,
}

/// Type of lead-in/lead-out path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LeadInType {
    /// No lead-in (pierce directly at contour).
    None,
    /// Straight line approach at an angle.
    Line,
    /// Circular arc tangent to the contour.
    Arc,
}

impl Default for LeadInConfig {
    fn default() -> Self {
        Self {
            lead_in_type: LeadInType::None,
            lead_in_length: 2.0,
            lead_out_length: 1.0,
            lead_in_angle: std::f64::consts::FRAC_PI_4, // 45 degrees
            arc_segments: 8,
        }
    }
}

/// Generated lead-in/lead-out paths for a pierce point.
#[derive(Debug, Clone)]
pub struct LeadInOut {
    /// Points forming the lead-in path (from start to pierce point).
    /// Empty if no lead-in.
    pub lead_in: Vec<(f64, f64)>,
    /// Points forming the lead-out path (from contour close to end).
    /// Empty if no lead-out.
    pub lead_out: Vec<(f64, f64)>,
}

/// Generates lead-in/lead-out paths for a pierce point on a contour.
///
/// The lead-in starts outside the part (for exterior contours) or inside the
/// hole (for interior contours) and ends at the pierce point. The lead-out
/// starts where the contour cut closes and extends away.
pub fn generate_lead_in_out(
    contour: &CutContour,
    pierce: &PierceSelection,
    config: &LeadInConfig,
) -> LeadInOut {
    if config.lead_in_type == LeadInType::None {
        return LeadInOut {
            lead_in: Vec::new(),
            lead_out: Vec::new(),
        };
    }

    let tangent = compute_tangent_at_pierce(contour, pierce);
    let outward_normal = compute_outward_normal(tangent, contour.contour_type, pierce.direction);

    let lead_in = match config.lead_in_type {
        LeadInType::None => Vec::new(),
        LeadInType::Line => generate_line_lead_in(
            pierce.point,
            tangent,
            outward_normal,
            config.lead_in_length,
            config.lead_in_angle,
        ),
        LeadInType::Arc => generate_arc_lead_in(
            pierce.point,
            tangent,
            outward_normal,
            config.lead_in_length,
            config.arc_segments,
        ),
    };

    let lead_out = if config.lead_out_length > 0.0 {
        generate_line_lead_out(
            pierce.end_point,
            tangent,
            outward_normal,
            config.lead_out_length,
        )
    } else {
        Vec::new()
    };

    LeadInOut { lead_in, lead_out }
}

/// Computes the tangent direction at the pierce point along the contour.
fn compute_tangent_at_pierce(contour: &CutContour, pierce: &PierceSelection) -> (f64, f64) {
    let n = contour.vertices.len();
    if n < 2 {
        return (1.0, 0.0); // Default direction
    }

    let i = pierce.vertex_index;
    let j = (i + 1) % n;
    let (ax, ay) = contour.vertices[i];
    let (bx, by) = contour.vertices[j];

    let dx = bx - ax;
    let dy = by - ay;
    let len = (dx * dx + dy * dy).sqrt();

    if len < 1e-12 {
        (1.0, 0.0)
    } else {
        (dx / len, dy / len)
    }
}

/// Computes the outward normal direction for lead-in placement.
///
/// For exterior contours (CCW): outward is to the left of the tangent
/// For interior contours (CW): outward (into waste) is to the right
fn compute_outward_normal(
    tangent: (f64, f64),
    contour_type: ContourType,
    direction: CutDirection,
) -> (f64, f64) {
    // Left normal of tangent: (-ty, tx)
    // Right normal of tangent: (ty, -tx)
    let (tx, ty) = tangent;

    match (contour_type, direction) {
        // Exterior CCW: outward is left (away from part)
        (ContourType::Exterior, CutDirection::Ccw) => (-ty, tx),
        // Exterior CW: outward is right
        (ContourType::Exterior, CutDirection::Cw) => (ty, -tx),
        // Interior CCW: into waste is right
        (ContourType::Interior, CutDirection::Ccw) => (ty, -tx),
        // Interior CW: into waste is left
        (ContourType::Interior, CutDirection::Cw) => (-ty, tx),
    }
}

/// Generates a straight-line lead-in path.
///
/// The lead-in starts at a point offset from the pierce point along a direction
/// that combines the outward normal and the negative tangent (approaching the
/// contour at an angle).
fn generate_line_lead_in(
    pierce: (f64, f64),
    tangent: (f64, f64),
    outward: (f64, f64),
    length: f64,
    angle: f64,
) -> Vec<(f64, f64)> {
    // Direction: rotate outward normal by -angle toward the approach direction
    // Approach = outward * cos(angle) + (-tangent) * sin(angle)
    let (cos_a, sin_a) = (angle.cos(), angle.sin());
    let approach_dx = outward.0 * cos_a - tangent.0 * sin_a;
    let approach_dy = outward.1 * cos_a - tangent.1 * sin_a;

    let start = (
        pierce.0 + approach_dx * length,
        pierce.1 + approach_dy * length,
    );

    vec![start, pierce]
}

/// Generates an arc lead-in path (quarter circle tangent to contour).
fn generate_arc_lead_in(
    pierce: (f64, f64),
    _tangent: (f64, f64),
    outward: (f64, f64),
    radius: f64,
    segments: usize,
) -> Vec<(f64, f64)> {
    let segments = segments.max(2);
    let center = (pierce.0 + outward.0 * radius, pierce.1 + outward.1 * radius);

    // Arc from 180 degrees (opposite of outward) to pierce point (0 degrees relative)
    // In the local frame where outward = (1, 0), pierce is at angle = PI
    let start_angle = std::f64::consts::PI;
    let end_angle = 0.0_f64; // NOT 2*PI, but 0.0 to go from start to pierce point
    let angle_span = end_angle - start_angle; // -PI = clockwise quarter circle
                                              // Actually we want a half-circle from start to pierce

    let mut points = Vec::with_capacity(segments + 1);
    for i in 0..=segments {
        let t = i as f64 / segments as f64;
        let angle = start_angle + angle_span * t;

        // Point on arc relative to center, in the frame of the outward normal
        let local_x = angle.cos() * radius;
        let local_y = angle.sin() * radius;

        // Transform to world coordinates using outward as the x-axis
        // outward = (ox, oy), perpendicular = (-oy, ox)
        let perp = (-outward.1, outward.0);
        let world_x = center.0 + local_x * outward.0 + local_y * perp.0;
        let world_y = center.1 + local_x * outward.1 + local_y * perp.1;

        points.push((world_x, world_y));
    }

    // Ensure the last point is exactly the pierce point
    if let Some(last) = points.last_mut() {
        *last = pierce;
    }

    points
}

/// Generates a straight-line lead-out path.
fn generate_line_lead_out(
    end_point: (f64, f64),
    tangent: (f64, f64),
    outward: (f64, f64),
    length: f64,
) -> Vec<(f64, f64)> {
    // Lead-out continues along the tangent then moves outward
    let exit = (
        end_point.0 + (tangent.0 + outward.0) * 0.5 * length,
        end_point.1 + (tangent.1 + outward.1) * 0.5 * length,
    );

    vec![end_point, exit]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contour::ContourType;

    fn make_square_contour(id: usize, size: f64, ct: ContourType) -> CutContour {
        let half = size / 2.0;
        CutContour {
            id,
            geometry_id: format!("part{}", id),
            instance: 0,
            contour_type: ct,
            vertices: vec![(-half, -half), (half, -half), (half, half), (-half, half)],
            perimeter: 4.0 * size,
            centroid: (0.0, 0.0),
        }
    }

    fn make_pierce(
        point: (f64, f64),
        vertex_index: usize,
        direction: CutDirection,
    ) -> PierceSelection {
        PierceSelection {
            point,
            vertex_index,
            direction,
            end_point: point,
        }
    }

    #[test]
    fn test_no_lead_in() {
        let contour = make_square_contour(0, 10.0, ContourType::Exterior);
        let pierce = make_pierce((-5.0, -5.0), 0, CutDirection::Ccw);
        let config = LeadInConfig {
            lead_in_type: LeadInType::None,
            ..Default::default()
        };

        let result = generate_lead_in_out(&contour, &pierce, &config);
        assert!(result.lead_in.is_empty());
        assert!(result.lead_out.is_empty());
    }

    #[test]
    fn test_line_lead_in_exterior() {
        let contour = make_square_contour(0, 10.0, ContourType::Exterior);
        let pierce = make_pierce((-5.0, -5.0), 0, CutDirection::Ccw);
        let config = LeadInConfig {
            lead_in_type: LeadInType::Line,
            lead_in_length: 3.0,
            lead_out_length: 1.5,
            ..Default::default()
        };

        let result = generate_lead_in_out(&contour, &pierce, &config);

        // Lead-in should have 2 points (start + pierce)
        assert_eq!(result.lead_in.len(), 2);
        // Last point should be the pierce point
        let last = result.lead_in.last().expect("has points");
        assert!((last.0 - (-5.0)).abs() < 1e-10);
        assert!((last.1 - (-5.0)).abs() < 1e-10);

        // Start point should be further away from the contour
        let start = result.lead_in[0];
        let dist_start = ((start.0 - (-5.0)).powi(2) + (start.1 - (-5.0)).powi(2)).sqrt();
        assert!(
            dist_start > 2.0,
            "Lead-in start should be offset from pierce: dist={}",
            dist_start
        );
    }

    #[test]
    fn test_line_lead_out() {
        let contour = make_square_contour(0, 10.0, ContourType::Exterior);
        let pierce = make_pierce((-5.0, -5.0), 0, CutDirection::Ccw);
        let config = LeadInConfig {
            lead_in_type: LeadInType::Line,
            lead_in_length: 3.0,
            lead_out_length: 2.0,
            ..Default::default()
        };

        let result = generate_lead_in_out(&contour, &pierce, &config);

        // Lead-out should have 2 points
        assert_eq!(result.lead_out.len(), 2);
        // First point is the end point (same as pierce for closed contour)
        let first = result.lead_out[0];
        assert!((first.0 - (-5.0)).abs() < 1e-10);
        assert!((first.1 - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_arc_lead_in() {
        let contour = make_square_contour(0, 10.0, ContourType::Exterior);
        let pierce = make_pierce((-5.0, -5.0), 0, CutDirection::Ccw);
        let config = LeadInConfig {
            lead_in_type: LeadInType::Arc,
            lead_in_length: 3.0,
            arc_segments: 8,
            ..Default::default()
        };

        let result = generate_lead_in_out(&contour, &pierce, &config);

        // Arc should have segments+1 points
        assert_eq!(result.lead_in.len(), 9);
        // Last point should be the pierce point
        let last = result.lead_in.last().expect("has points");
        assert!((last.0 - (-5.0)).abs() < 1e-10);
        assert!((last.1 - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_interior_lead_in_direction() {
        // Interior hole: lead-in should go into the waste (inside the hole)
        let contour = make_square_contour(0, 10.0, ContourType::Interior);
        let pierce = make_pierce((5.0, 0.0), 1, CutDirection::Cw);
        let config = LeadInConfig {
            lead_in_type: LeadInType::Line,
            lead_in_length: 3.0,
            lead_out_length: 0.0,
            ..Default::default()
        };

        let result = generate_lead_in_out(&contour, &pierce, &config);
        assert_eq!(result.lead_in.len(), 2);

        // For interior CW on the right edge going up,
        // outward (into waste) should be to the left (toward center)
        let start = result.lead_in[0];
        // Start should be to the left of the pierce (toward interior)
        assert!(
            start.0 < 5.0,
            "Interior lead-in start should be inside hole: x={}",
            start.0
        );
    }

    #[test]
    fn test_zero_lead_out_length() {
        let contour = make_square_contour(0, 10.0, ContourType::Exterior);
        let pierce = make_pierce((-5.0, -5.0), 0, CutDirection::Ccw);
        let config = LeadInConfig {
            lead_in_type: LeadInType::Line,
            lead_in_length: 3.0,
            lead_out_length: 0.0,
            ..Default::default()
        };

        let result = generate_lead_in_out(&contour, &pierce, &config);
        assert_eq!(result.lead_in.len(), 2);
        assert!(result.lead_out.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = LeadInConfig::default();
        assert_eq!(config.lead_in_type, LeadInType::None);
        assert!((config.lead_in_length - 2.0).abs() < 1e-10);
        assert!((config.lead_out_length - 1.0).abs() < 1e-10);
        assert!((config.lead_in_angle - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }
}
