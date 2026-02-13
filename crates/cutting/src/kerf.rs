//! Kerf (cutting tool width) compensation.
//!
//! Applies polygon offset to contour vertices to compensate for the material
//! removed by the cutting tool. The cut path follows the offset contour so
//! that the final part dimensions match the original design.
//!
//! # Offset Direction
//!
//! - **Exterior contours**: Offset outward by `+kerf/2` (tool cuts outside the part)
//! - **Interior holes**: Offset inward by `-kerf/2` (tool cuts inside the hole)
//!
//! # References
//!
//! - Standard practice in CAD/CAM: half-kerf-width offset from design edge

use crate::config::CuttingConfig;
use crate::contour::{ContourType, CutContour};

/// Result of kerf compensation for a single contour.
#[derive(Debug)]
pub enum KerfResult {
    /// Successfully compensated contour with new vertices.
    Compensated(CutContour),
    /// Contour collapsed (too small for the kerf width). Should be skipped.
    Collapsed {
        /// The original contour ID.
        contour_id: usize,
        /// Reason for collapse.
        reason: String,
    },
}

/// Applies kerf compensation to a set of cut contours.
///
/// For each contour:
/// - Exterior: offset by `+kerf_width/2` (outward)
/// - Interior: offset by `-kerf_width/2` (inward)
///
/// Contours that collapse (too small for the offset) are reported as
/// `KerfResult::Collapsed` rather than silently dropped.
///
/// Returns an empty vec if `config.kerf_width` is zero or negative.
pub fn apply_kerf_compensation(
    contours: &[CutContour],
    config: &CuttingConfig,
) -> Vec<KerfResult> {
    let half_kerf = config.kerf_width / 2.0;

    if half_kerf <= 0.0 {
        // No compensation needed — return contours as-is
        return contours.iter().map(|c| KerfResult::Compensated(c.clone())).collect();
    }

    contours
        .iter()
        .map(|contour| compensate_contour(contour, half_kerf))
        .collect()
}

/// Applies kerf compensation to a single contour.
fn compensate_contour(contour: &CutContour, half_kerf: f64) -> KerfResult {
    let offset_distance = match contour.contour_type {
        ContourType::Exterior => half_kerf,  // Outward expansion
        ContourType::Interior => -half_kerf, // Inward shrinkage
    };

    let offset_results = u_nesting_core::geom::offset::offset_polygon(
        &contour.vertices,
        offset_distance,
    );

    if offset_results.is_empty() {
        return KerfResult::Collapsed {
            contour_id: contour.id,
            reason: format!(
                "contour {} collapsed with kerf offset {:.4}",
                contour.id, offset_distance
            ),
        };
    }

    // Use the largest polygon (by vertex count) from the offset results.
    // Multiple polygons can arise from self-intersection resolution, but
    // for kerf compensation we want the main contour.
    let best = offset_results
        .into_iter()
        .max_by_key(|ring| ring.len())
        .expect("offset_results is non-empty after is_empty check");

    let new_perimeter = u_nesting_core::geom::polygon::perimeter(&best);
    let new_centroid = u_nesting_core::geom::polygon::centroid(&best)
        .unwrap_or(contour.centroid);

    KerfResult::Compensated(CutContour {
        id: contour.id,
        geometry_id: contour.geometry_id.clone(),
        instance: contour.instance,
        contour_type: contour.contour_type,
        vertices: best,
        perimeter: new_perimeter,
        centroid: new_centroid,
    })
}

/// Filters kerf results, keeping only successfully compensated contours.
///
/// Collapsed contours are logged at warn level and excluded.
pub fn filter_compensated(results: Vec<KerfResult>) -> Vec<CutContour> {
    results
        .into_iter()
        .filter_map(|r| match r {
            KerfResult::Compensated(c) => Some(c),
            KerfResult::Collapsed { contour_id, reason } => {
                log::warn!("Kerf compensation: contour {} collapsed — {}", contour_id, reason);
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square(id: usize, size: f64, ct: ContourType) -> CutContour {
        let half = size / 2.0;
        CutContour {
            id,
            geometry_id: format!("part{}", id),
            instance: 0,
            contour_type: ct,
            vertices: vec![
                (-half, -half),
                (half, -half),
                (half, half),
                (-half, half),
            ],
            perimeter: 4.0 * size,
            centroid: (0.0, 0.0),
        }
    }

    #[test]
    fn test_zero_kerf_returns_original() {
        let contours = vec![make_square(0, 10.0, ContourType::Exterior)];
        let config = CuttingConfig::new().with_kerf_width(0.0);
        let results = apply_kerf_compensation(&contours, &config);

        assert_eq!(results.len(), 1);
        if let KerfResult::Compensated(c) = &results[0] {
            assert_eq!(c.vertices.len(), 4);
            // Vertices should be unchanged
            assert!((c.vertices[0].0 - (-5.0)).abs() < 1e-10);
        } else {
            panic!("expected Compensated");
        }
    }

    #[test]
    fn test_exterior_offset_outward() {
        let contours = vec![make_square(0, 10.0, ContourType::Exterior)];
        let config = CuttingConfig::new().with_kerf_width(1.0); // half = 0.5

        let results = apply_kerf_compensation(&contours, &config);
        assert_eq!(results.len(), 1);

        if let KerfResult::Compensated(c) = &results[0] {
            // Exterior should be expanded: area should increase
            let original_area = u_nesting_core::geom::polygon::signed_area(
                &make_square(0, 10.0, ContourType::Exterior).vertices,
            ).abs();
            let new_area = u_nesting_core::geom::polygon::signed_area(&c.vertices).abs();
            assert!(
                new_area > original_area,
                "Exterior kerf should expand: new_area={} > original_area={}",
                new_area, original_area
            );
        } else {
            panic!("expected Compensated");
        }
    }

    #[test]
    fn test_interior_offset_inward() {
        let contours = vec![make_square(0, 10.0, ContourType::Interior)];
        let config = CuttingConfig::new().with_kerf_width(1.0); // half = 0.5

        let results = apply_kerf_compensation(&contours, &config);
        assert_eq!(results.len(), 1);

        if let KerfResult::Compensated(c) = &results[0] {
            // Interior should shrink: area should decrease
            let original_area = u_nesting_core::geom::polygon::signed_area(
                &make_square(0, 10.0, ContourType::Interior).vertices,
            ).abs();
            let new_area = u_nesting_core::geom::polygon::signed_area(&c.vertices).abs();
            assert!(
                new_area < original_area,
                "Interior kerf should shrink: new_area={} < original_area={}",
                new_area, original_area
            );
        } else {
            panic!("expected Compensated");
        }
    }

    #[test]
    fn test_small_contour_collapses() {
        // 2x2 square with kerf_width=5 → half=2.5, should collapse
        let contours = vec![make_square(0, 2.0, ContourType::Interior)];
        let config = CuttingConfig::new().with_kerf_width(5.0);

        let results = apply_kerf_compensation(&contours, &config);
        assert_eq!(results.len(), 1);

        match &results[0] {
            KerfResult::Collapsed { contour_id, .. } => {
                assert_eq!(*contour_id, 0);
            }
            KerfResult::Compensated(_) => panic!("expected Collapsed for tiny contour"),
        }
    }

    #[test]
    fn test_filter_compensated() {
        let results = vec![
            KerfResult::Compensated(make_square(0, 10.0, ContourType::Exterior)),
            KerfResult::Collapsed {
                contour_id: 1,
                reason: "test collapse".to_string(),
            },
            KerfResult::Compensated(make_square(2, 10.0, ContourType::Exterior)),
        ];

        let filtered = filter_compensated(results);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id, 0);
        assert_eq!(filtered[1].id, 2);
    }

    #[test]
    fn test_mixed_contour_types() {
        let contours = vec![
            make_square(0, 20.0, ContourType::Exterior),
            make_square(1, 8.0, ContourType::Interior),
        ];
        let config = CuttingConfig::new().with_kerf_width(0.5);

        let results = apply_kerf_compensation(&contours, &config);
        let compensated = filter_compensated(results);
        assert_eq!(compensated.len(), 2);

        // Exterior expanded
        let ext = &compensated[0];
        let ext_area = u_nesting_core::geom::polygon::signed_area(&ext.vertices).abs();
        assert!(ext_area > 400.0); // Original 20x20 = 400

        // Interior shrunk
        let int = &compensated[1];
        let int_area = u_nesting_core::geom::polygon::signed_area(&int.vertices).abs();
        assert!(int_area < 64.0); // Original 8x8 = 64
    }

    #[test]
    fn test_perimeter_updated() {
        let contours = vec![make_square(0, 10.0, ContourType::Exterior)];
        let config = CuttingConfig::new().with_kerf_width(1.0);

        let results = apply_kerf_compensation(&contours, &config);
        if let KerfResult::Compensated(c) = &results[0] {
            // Expanded square perimeter should be larger
            assert!(c.perimeter > 40.0, "Perimeter should increase: {}", c.perimeter);
        } else {
            panic!("expected Compensated");
        }
    }
}
