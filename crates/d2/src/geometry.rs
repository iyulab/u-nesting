//! 2D geometry types.

use geo::{Area, Centroid, ConvexHull, Coord, LineString, Polygon as GeoPolygon};
use u_nesting_core::geometry::{Geometry, Geometry2DExt, GeometryId, RotationConstraint};
use u_nesting_core::transform::AABB2D;
use u_nesting_core::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 2D polygon geometry that can be nested.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Geometry2D {
    /// Unique identifier.
    id: GeometryId,

    /// Outer boundary of the polygon.
    exterior: Vec<(f64, f64)>,

    /// Interior holes (if any).
    holes: Vec<Vec<(f64, f64)>>,

    /// Number of copies to place.
    quantity: usize,

    /// Rotation constraint.
    rotation_constraint: RotationConstraint<f64>,

    /// Whether the geometry can be flipped (mirrored).
    allow_flip: bool,

    /// Placement priority (higher = placed first).
    priority: i32,

    /// Cached area.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_area: Option<f64>,

    /// Cached convex hull.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_convex_hull: Option<Vec<(f64, f64)>>,

    /// Cached perimeter.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_perimeter: Option<f64>,

    /// Cached convexity flag.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_is_convex: Option<bool>,
}

impl Geometry2D {
    /// Creates a new 2D geometry with the given ID.
    pub fn new(id: impl Into<GeometryId>) -> Self {
        Self {
            id: id.into(),
            exterior: Vec::new(),
            holes: Vec::new(),
            quantity: 1,
            rotation_constraint: RotationConstraint::None,
            allow_flip: false,
            priority: 0,
            cached_area: None,
            cached_convex_hull: None,
            cached_perimeter: None,
            cached_is_convex: None,
        }
    }

    /// Sets the polygon from a list of (x, y) vertices.
    pub fn with_polygon(mut self, vertices: Vec<(f64, f64)>) -> Self {
        self.exterior = vertices;
        self.clear_cache();
        self
    }

    /// Adds an interior hole.
    pub fn with_hole(mut self, vertices: Vec<(f64, f64)>) -> Self {
        self.holes.push(vertices);
        self.clear_cache();
        self
    }

    /// Sets the quantity to place.
    pub fn with_quantity(mut self, n: usize) -> Self {
        self.quantity = n;
        self
    }

    /// Sets the allowed rotation angles in degrees.
    pub fn with_rotations_deg(mut self, angles: Vec<f64>) -> Self {
        let radians: Vec<f64> = angles.into_iter().map(|a| a.to_radians()).collect();
        self.rotation_constraint = RotationConstraint::Discrete(radians);
        self
    }

    /// Sets the allowed rotation angles in radians.
    pub fn with_rotations(mut self, angles: Vec<f64>) -> Self {
        self.rotation_constraint = RotationConstraint::Discrete(angles);
        self
    }

    /// Sets the rotation constraint.
    pub fn with_rotation_constraint(mut self, constraint: RotationConstraint<f64>) -> Self {
        self.rotation_constraint = constraint;
        self
    }

    /// Allows flipping (mirroring) the geometry.
    pub fn with_flip(mut self, allow: bool) -> Self {
        self.allow_flip = allow;
        self
    }

    /// Sets the placement priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Creates a rectangular geometry.
    pub fn rectangle(id: impl Into<GeometryId>, width: f64, height: f64) -> Self {
        Self::new(id).with_polygon(vec![
            (0.0, 0.0),
            (width, 0.0),
            (width, height),
            (0.0, height),
        ])
    }

    /// Creates a circle approximation with n vertices.
    pub fn circle(id: impl Into<GeometryId>, radius: f64, n: usize) -> Self {
        let n = n.max(8);
        let step = std::f64::consts::TAU / n as f64;
        let vertices: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let angle = i as f64 * step;
                (radius * angle.cos() + radius, radius * angle.sin() + radius)
            })
            .collect();
        Self::new(id).with_polygon(vertices)
    }

    /// Creates an L-shaped geometry.
    pub fn l_shape(
        id: impl Into<GeometryId>,
        width: f64,
        height: f64,
        notch_width: f64,
        notch_height: f64,
    ) -> Self {
        Self::new(id).with_polygon(vec![
            (0.0, 0.0),
            (width, 0.0),
            (width, notch_height),
            (notch_width, notch_height),
            (notch_width, height),
            (0.0, height),
        ])
    }

    /// Returns the exterior vertices.
    pub fn exterior(&self) -> &[(f64, f64)] {
        &self.exterior
    }

    /// Returns the allowed rotation angles (for compatibility).
    pub fn rotations(&self) -> Vec<f64> {
        self.rotation_constraint.angles()
    }

    /// Returns whether flipping is allowed.
    pub fn allow_flip(&self) -> bool {
        self.allow_flip
    }

    /// Converts to a geo crate Polygon.
    pub fn to_geo_polygon(&self) -> GeoPolygon<f64> {
        let exterior = LineString::from(
            self.exterior
                .iter()
                .map(|&(x, y)| Coord { x, y })
                .collect::<Vec<_>>(),
        );

        let holes: Vec<LineString<f64>> = self
            .holes
            .iter()
            .map(|hole| {
                LineString::from(
                    hole.iter()
                        .map(|&(x, y)| Coord { x, y })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        GeoPolygon::new(exterior, holes)
    }

    /// Clears all cached values.
    fn clear_cache(&mut self) {
        self.cached_area = None;
        self.cached_convex_hull = None;
        self.cached_perimeter = None;
        self.cached_is_convex = None;
    }

    /// Calculates the area of the polygon.
    fn calculate_area(&self) -> f64 {
        self.to_geo_polygon().unsigned_area()
    }

    /// Calculates the perimeter of the polygon.
    fn calculate_perimeter(&self) -> f64 {
        use geo::{Euclidean, Length};
        let poly = self.to_geo_polygon();
        let mut perim = poly.exterior().length::<Euclidean>();
        for hole in poly.interiors() {
            perim += hole.length::<Euclidean>();
        }
        perim
    }

    /// Calculates the convex hull.
    fn calculate_convex_hull(&self) -> Vec<(f64, f64)> {
        let poly = self.to_geo_polygon();
        let hull = poly.convex_hull();
        hull.exterior().points().map(|p| (p.x(), p.y())).collect()
    }

    /// Checks if the polygon is convex.
    fn calculate_is_convex(&self) -> bool {
        if self.exterior.len() < 3 || !self.holes.is_empty() {
            return false;
        }

        // A polygon is convex if all cross products of consecutive edge pairs
        // have the same sign
        let n = self.exterior.len();
        let mut sign = 0i32;

        for i in 0..n {
            let (x1, y1) = self.exterior[i];
            let (x2, y2) = self.exterior[(i + 1) % n];
            let (x3, y3) = self.exterior[(i + 2) % n];

            let cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2);

            if cross.abs() > 1e-10 {
                let current_sign = if cross > 0.0 { 1 } else { -1 };
                if sign == 0 {
                    sign = current_sign;
                } else if sign != current_sign {
                    return false;
                }
            }
        }

        true
    }
}

impl Geometry for Geometry2D {
    type Scalar = f64;

    fn id(&self) -> &GeometryId {
        &self.id
    }

    fn quantity(&self) -> usize {
        self.quantity
    }

    fn measure(&self) -> f64 {
        if let Some(area) = self.cached_area {
            area
        } else {
            self.calculate_area()
        }
    }

    fn aabb(&self) -> ([f64; 2], [f64; 2]) {
        let (min, max) = self.aabb_vec();
        ([min[0], min[1]], [max[0], max[1]])
    }

    fn aabb_vec(&self) -> (Vec<f64>, Vec<f64>) {
        if self.exterior.is_empty() {
            return (vec![0.0, 0.0], vec![0.0, 0.0]);
        }

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for &(x, y) in &self.exterior {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        (vec![min_x, min_y], vec![max_x, max_y])
    }

    fn centroid(&self) -> Vec<f64> {
        if let Some(c) = self.to_geo_polygon().centroid() {
            vec![c.x(), c.y()]
        } else {
            vec![0.0, 0.0]
        }
    }

    fn validate(&self) -> Result<()> {
        if self.exterior.len() < 3 {
            return Err(Error::InvalidGeometry(format!(
                "Polygon '{}' must have at least 3 vertices",
                self.id
            )));
        }

        if self.quantity == 0 {
            return Err(Error::InvalidGeometry(format!(
                "Quantity for '{}' must be at least 1",
                self.id
            )));
        }

        // Check for self-intersection could be added here

        Ok(())
    }

    fn rotation_constraint(&self) -> &RotationConstraint<f64> {
        &self.rotation_constraint
    }

    fn allow_mirror(&self) -> bool {
        self.allow_flip
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

impl Geometry2DExt for Geometry2D {
    fn aabb_2d(&self) -> AABB2D<f64> {
        let (min, max) = self.aabb_vec();
        AABB2D::new(min[0], min[1], max[0], max[1])
    }

    fn outer_ring(&self) -> &[(f64, f64)] {
        &self.exterior
    }

    fn holes(&self) -> &[Vec<(f64, f64)>] {
        &self.holes
    }

    fn is_convex(&self) -> bool {
        if let Some(is_convex) = self.cached_is_convex {
            is_convex
        } else {
            self.calculate_is_convex()
        }
    }

    fn convex_hull(&self) -> Vec<(f64, f64)> {
        if let Some(ref hull) = self.cached_convex_hull {
            hull.clone()
        } else {
            self.calculate_convex_hull()
        }
    }

    fn perimeter(&self) -> f64 {
        if let Some(perim) = self.cached_perimeter {
            perim
        } else {
            self.calculate_perimeter()
        }
    }
}

impl Geometry2D {
    /// Computes the AABB of the geometry at a given rotation angle (in radians).
    ///
    /// Returns (min, max) as ([min_x, min_y], [max_x, max_y])
    pub fn aabb_at_rotation(&self, rotation: f64) -> ([f64; 2], [f64; 2]) {
        if rotation.abs() < 1e-10 {
            return self.aabb();
        }

        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in &self.exterior {
            let rx = x * cos_r - y * sin_r;
            let ry = x * sin_r + y * cos_r;
            min_x = min_x.min(rx);
            min_y = min_y.min(ry);
            max_x = max_x.max(rx);
            max_y = max_y.max(ry);
        }

        ([min_x, min_y], [max_x, max_y])
    }

    /// Returns the width and height of the AABB at a given rotation.
    pub fn dimensions_at_rotation(&self, rotation: f64) -> (f64, f64) {
        let (min, max) = self.aabb_at_rotation(rotation);
        (max[0] - min[0], max[1] - min[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rectangle_area() {
        let rect = Geometry2D::rectangle("R1", 10.0, 5.0);
        assert_relative_eq!(rect.measure(), 50.0, epsilon = 0.001);
    }

    #[test]
    fn test_polygon_with_hole() {
        let poly = Geometry2D::new("P1")
            .with_polygon(vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
            .with_hole(vec![(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)]);

        // Area = 100*100 - 50*50 = 10000 - 2500 = 7500
        assert_relative_eq!(poly.measure(), 7500.0, epsilon = 0.001);
    }

    #[test]
    fn test_aabb() {
        let poly = Geometry2D::new("P1").with_polygon(vec![
            (10.0, 20.0),
            (50.0, 20.0),
            (50.0, 80.0),
            (10.0, 80.0),
        ]);

        let aabb = poly.aabb_2d();
        assert_relative_eq!(aabb.min_x, 10.0);
        assert_relative_eq!(aabb.min_y, 20.0);
        assert_relative_eq!(aabb.max_x, 50.0);
        assert_relative_eq!(aabb.max_y, 80.0);
    }

    #[test]
    fn test_rectangle_is_convex() {
        let rect = Geometry2D::rectangle("R1", 10.0, 10.0);
        assert!(rect.is_convex());
    }

    #[test]
    fn test_l_shape_is_not_convex() {
        let l = Geometry2D::l_shape("L1", 20.0, 20.0, 10.0, 10.0);
        assert!(!l.is_convex());
    }

    #[test]
    fn test_convex_hull() {
        let l = Geometry2D::l_shape("L1", 20.0, 20.0, 10.0, 10.0);
        let hull = l.convex_hull();
        // Convex hull of L-shape should be a quadrilateral
        assert!(hull.len() >= 4);
    }

    #[test]
    fn test_centroid() {
        let rect = Geometry2D::rectangle("R1", 10.0, 10.0);
        let centroid = rect.centroid();
        assert_relative_eq!(centroid[0], 5.0, epsilon = 0.001);
        assert_relative_eq!(centroid[1], 5.0, epsilon = 0.001);
    }

    #[test]
    fn test_perimeter() {
        let rect = Geometry2D::rectangle("R1", 10.0, 5.0);
        assert_relative_eq!(rect.perimeter(), 30.0, epsilon = 0.001);
    }

    #[test]
    fn test_validation() {
        let valid = Geometry2D::rectangle("R1", 10.0, 10.0);
        assert!(valid.validate().is_ok());

        let invalid = Geometry2D::new("P1").with_polygon(vec![(0.0, 0.0), (1.0, 0.0)]);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_circle() {
        let circle = Geometry2D::circle("C1", 10.0, 32);
        let area = circle.measure();
        let expected = std::f64::consts::PI * 10.0 * 10.0;
        // Circle approximation should be close to actual area
        assert_relative_eq!(area, expected, epsilon = 5.0);
    }
}
