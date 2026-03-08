//! Contour extraction from nesting results.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use u_nesting_core::geometry::{Geometry, Geometry2DExt};
use u_nesting_core::SolveResult;

/// Unique identifier for a cut contour.
pub type ContourId = usize;

/// Type of contour in the cutting context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ContourType {
    /// External boundary of a part.
    Exterior,
    /// Internal hole within a part.
    Interior,
}

/// A single contour to be cut, extracted from nesting results.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CutContour {
    /// Unique ID for this contour.
    pub id: ContourId,
    /// ID of the source geometry.
    pub geometry_id: String,
    /// Instance index of the placed geometry (0-based).
    pub instance: usize,
    /// Whether this is an exterior or interior contour.
    pub contour_type: ContourType,
    /// Vertices in world coordinates (after placement transform).
    pub vertices: Vec<(f64, f64)>,
    /// Perimeter of this contour.
    pub perimeter: f64,
    /// Centroid of this contour.
    pub centroid: (f64, f64),
}

/// Extracts cut contours from a nesting solve result.
///
/// For each placed geometry instance, extracts the exterior contour and
/// any interior hole contours. All vertices are transformed to world
/// coordinates using the placement's position and rotation.
///
/// # Arguments
///
/// * `solve_result` - The nesting solve result with placements
/// * `geometries` - The original geometry definitions (must match geometry_ids in placements)
///
/// # Returns
///
/// A vector of `CutContour` with unique IDs, in order:
/// exterior and holes for each placed instance.
pub fn extract_contours<G: Geometry2DExt<Scalar = f64> + Geometry<Scalar = f64>>(
    solve_result: &SolveResult<f64>,
    geometries: &[G],
) -> Vec<CutContour> {
    let mut contours = Vec::new();
    let mut next_id: ContourId = 0;

    for placement in &solve_result.placements {
        // Find matching geometry
        let geom = match geometries.iter().find(|g| *g.id() == placement.geometry_id) {
            Some(g) => g,
            None => continue,
        };

        // Build transform from placement
        let transform = placement.to_transform_2d();

        // Extract exterior contour
        let exterior_world = transform.transform_points(geom.outer_ring());
        let exterior_perim = u_nesting_core::geom::polygon::perimeter(&exterior_world);
        let exterior_centroid =
            u_nesting_core::geom::polygon::centroid(&exterior_world).unwrap_or((0.0, 0.0));

        contours.push(CutContour {
            id: next_id,
            geometry_id: placement.geometry_id.clone(),
            instance: placement.instance,
            contour_type: ContourType::Exterior,
            vertices: exterior_world,
            perimeter: exterior_perim,
            centroid: exterior_centroid,
        });
        next_id += 1;

        // Extract interior hole contours
        for hole in geom.holes() {
            let hole_world = transform.transform_points(hole);
            let hole_perim = u_nesting_core::geom::polygon::perimeter(&hole_world);
            let hole_centroid =
                u_nesting_core::geom::polygon::centroid(&hole_world).unwrap_or((0.0, 0.0));

            contours.push(CutContour {
                id: next_id,
                geometry_id: placement.geometry_id.clone(),
                instance: placement.instance,
                contour_type: ContourType::Interior,
                vertices: hole_world,
                perimeter: hole_perim,
                centroid: hole_centroid,
            });
            next_id += 1;
        }
    }

    contours
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_type_equality() {
        assert_eq!(ContourType::Exterior, ContourType::Exterior);
        assert_ne!(ContourType::Exterior, ContourType::Interior);
    }

    #[test]
    fn test_cut_contour_creation() {
        let contour = CutContour {
            id: 0,
            geometry_id: "rect1".to_string(),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            perimeter: 40.0,
            centroid: (5.0, 5.0),
        };
        assert_eq!(contour.id, 0);
        assert_eq!(contour.contour_type, ContourType::Exterior);
        assert_eq!(contour.vertices.len(), 4);
    }
}
