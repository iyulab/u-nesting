//! Spatial indexing for 3D collision detection using AABB trees.
//!
//! This module provides efficient broad-phase collision detection for the packing
//! algorithm using axis-aligned bounding box (AABB) trees.

use crate::geometry::Geometry3D;
use nalgebra::Vector3;
use u_nesting_core::geometry::Geometry;

/// Simple 3D AABB (Axis-Aligned Bounding Box).
#[derive(Debug, Clone, Copy)]
pub struct Aabb3D {
    /// Minimum corner (x, y, z)
    pub min: [f64; 3],
    /// Maximum corner (x, y, z)
    pub max: [f64; 3],
}

impl Aabb3D {
    /// Creates a new AABB.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Creates an AABB from position and dimensions.
    pub fn from_position_and_size(position: [f64; 3], size: [f64; 3]) -> Self {
        Self {
            min: position,
            max: [
                position[0] + size[0],
                position[1] + size[1],
                position[2] + size[2],
            ],
        }
    }

    /// Checks if this AABB intersects with another AABB.
    pub fn intersects(&self, other: &Aabb3D) -> bool {
        self.min[0] < other.max[0]
            && self.max[0] > other.min[0]
            && self.min[1] < other.max[1]
            && self.max[1] > other.min[1]
            && self.min[2] < other.max[2]
            && self.max[2] > other.min[2]
    }

    /// Checks if this AABB contains a point.
    pub fn contains_point(&self, point: [f64; 3]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
            && point[2] >= self.min[2]
            && point[2] <= self.max[2]
    }

    /// Checks if this AABB is fully contained within another AABB.
    pub fn is_within(&self, other: &Aabb3D) -> bool {
        self.min[0] >= other.min[0]
            && self.min[1] >= other.min[1]
            && self.min[2] >= other.min[2]
            && self.max[0] <= other.max[0]
            && self.max[1] <= other.max[1]
            && self.max[2] <= other.max[2]
    }

    /// Expands this AABB by a margin on all sides.
    pub fn expand(&self, margin: f64) -> Self {
        Self {
            min: [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            max: [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        }
    }

    /// Returns the volume of this AABB.
    pub fn volume(&self) -> f64 {
        (self.max[0] - self.min[0]) * (self.max[1] - self.min[1]) * (self.max[2] - self.min[2])
    }
}

/// An entry in the 3D spatial index representing a placed geometry.
#[derive(Debug, Clone)]
pub struct SpatialEntry3D {
    /// Index of the geometry in the placed list
    pub index: usize,
    /// Geometry ID
    pub id: String,
    /// Axis-aligned bounding box
    pub aabb: Aabb3D,
}

impl SpatialEntry3D {
    /// Creates a new spatial entry.
    pub fn new(index: usize, id: String, aabb: Aabb3D) -> Self {
        Self { index, id, aabb }
    }

    /// Creates a spatial entry from a placed geometry.
    pub fn from_placed(
        index: usize,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
    ) -> Self {
        let aabb = compute_transformed_aabb(geometry, position, orientation);
        Self {
            index,
            id: geometry.id().clone(),
            aabb,
        }
    }
}

/// 3D spatial index using a simple AABB list with efficient queries.
///
/// For small to medium numbers of objects, a simple list with AABB tests
/// outperforms tree-based structures due to cache locality.
/// For larger numbers (>1000), a BVH would be more efficient.
#[derive(Debug)]
pub struct SpatialIndex3D {
    entries: Vec<SpatialEntry3D>,
}

impl SpatialIndex3D {
    /// Creates a new empty spatial index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Creates a spatial index with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Inserts a new entry into the spatial index.
    pub fn insert(&mut self, entry: SpatialEntry3D) {
        self.entries.push(entry);
    }

    /// Inserts a placed geometry into the spatial index.
    pub fn insert_geometry(
        &mut self,
        index: usize,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
    ) {
        let entry = SpatialEntry3D::from_placed(index, geometry, position, orientation);
        self.insert(entry);
    }

    /// Returns the number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all entries from the index.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Finds all entries whose bounding boxes intersect with the given AABB.
    pub fn query_aabb(&self, query_aabb: &Aabb3D) -> Vec<&SpatialEntry3D> {
        self.entries
            .iter()
            .filter(|entry| entry.aabb.intersects(query_aabb))
            .collect()
    }

    /// Finds all entries that potentially intersect with a geometry at the given position.
    pub fn query_geometry(
        &self,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
    ) -> Vec<&SpatialEntry3D> {
        let query_aabb = compute_transformed_aabb(geometry, position, orientation);
        self.query_aabb(&query_aabb)
    }

    /// Finds all entries that potentially intersect with the given geometry AABB
    /// expanded by a margin (for spacing).
    pub fn query_with_margin(
        &self,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
        margin: f64,
    ) -> Vec<&SpatialEntry3D> {
        let base_aabb = compute_transformed_aabb(geometry, position, orientation);
        let expanded_aabb = base_aabb.expand(margin);
        self.query_aabb(&expanded_aabb)
    }

    /// Returns an iterator over all entries in the index.
    pub fn iter(&self) -> impl Iterator<Item = &SpatialEntry3D> {
        self.entries.iter()
    }

    /// Returns the indices of potentially colliding geometries for a query geometry.
    pub fn get_potential_collisions(
        &self,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
        spacing: f64,
    ) -> Vec<usize> {
        self.query_with_margin(geometry, position, orientation, spacing)
            .iter()
            .map(|entry| entry.index)
            .collect()
    }

    /// Checks if a geometry at the given position would collide with any existing entries.
    pub fn has_collision(
        &self,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
        spacing: f64,
    ) -> bool {
        let base_aabb = compute_transformed_aabb(geometry, position, orientation);
        let expanded_aabb = base_aabb.expand(spacing);

        self.entries
            .iter()
            .any(|entry| entry.aabb.intersects(&expanded_aabb))
    }

    /// Checks if a geometry at the given position fits within the boundary and
    /// doesn't collide with any existing entries.
    pub fn can_place(
        &self,
        geometry: &Geometry3D,
        position: Vector3<f64>,
        orientation: usize,
        boundary_aabb: &Aabb3D,
        spacing: f64,
    ) -> bool {
        let geom_aabb = compute_transformed_aabb(geometry, position, orientation);

        // Check if geometry fits within boundary
        if !geom_aabb.is_within(boundary_aabb) {
            return false;
        }

        // Check for collisions with existing entries
        !self.has_collision(geometry, position, orientation, spacing)
    }
}

impl Default for SpatialIndex3D {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the transformed AABB of a geometry at a given position with the specified orientation.
pub fn compute_transformed_aabb(
    geometry: &Geometry3D,
    position: Vector3<f64>,
    orientation: usize,
) -> Aabb3D {
    let dims = geometry.dimensions_for_orientation(orientation);

    Aabb3D::from_position_and_size(
        [position.x, position.y, position.z],
        [dims.x, dims.y, dims.z],
    )
}

/// Creates a boundary AABB with the given dimensions and margin.
pub fn boundary_aabb(width: f64, depth: f64, height: f64, margin: f64) -> Aabb3D {
    Aabb3D::new(
        [margin, margin, margin],
        [width - margin, depth - margin, height - margin],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_intersects() {
        let a = Aabb3D::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = Aabb3D::new([5.0, 5.0, 5.0], [15.0, 15.0, 15.0]);
        let c = Aabb3D::new([20.0, 20.0, 20.0], [30.0, 30.0, 30.0]);

        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
        assert!(!c.intersects(&a));
    }

    #[test]
    fn test_aabb_is_within() {
        let outer = Aabb3D::new([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]);
        let inner = Aabb3D::new([10.0, 10.0, 10.0], [20.0, 20.0, 20.0]);
        let partial = Aabb3D::new([90.0, 90.0, 90.0], [110.0, 110.0, 110.0]);

        assert!(inner.is_within(&outer));
        assert!(!partial.is_within(&outer));
        assert!(!outer.is_within(&inner));
    }

    #[test]
    fn test_spatial_index_new() {
        let index = SpatialIndex3D::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_index_insert() {
        let mut index = SpatialIndex3D::new();
        let aabb = Aabb3D::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let entry = SpatialEntry3D::new(0, "test".to_string(), aabb);
        index.insert(entry);

        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_spatial_index_query_aabb() {
        let mut index = SpatialIndex3D::new();

        // Insert three non-overlapping boxes
        index.insert(SpatialEntry3D::new(
            0,
            "b1".to_string(),
            Aabb3D::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        ));
        index.insert(SpatialEntry3D::new(
            1,
            "b2".to_string(),
            Aabb3D::new([20.0, 0.0, 0.0], [30.0, 10.0, 10.0]),
        ));
        index.insert(SpatialEntry3D::new(
            2,
            "b3".to_string(),
            Aabb3D::new([0.0, 20.0, 0.0], [10.0, 30.0, 10.0]),
        ));

        // Query overlapping with b1 only
        let query = Aabb3D::new([5.0, 5.0, 5.0], [15.0, 15.0, 15.0]);
        let results = index.query_aabb(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);

        // Query overlapping with b1 and b2
        let query = Aabb3D::new([5.0, 0.0, 0.0], [25.0, 10.0, 10.0]);
        let results = index.query_aabb(&query);
        assert_eq!(results.len(), 2);

        // Query overlapping with nothing
        let query = Aabb3D::new([50.0, 50.0, 50.0], [60.0, 60.0, 60.0]);
        let results = index.query_aabb(&query);
        assert!(results.is_empty());

        // Query overlapping with all
        let query = Aabb3D::new([-10.0, -10.0, -10.0], [40.0, 40.0, 40.0]);
        let results = index.query_aabb(&query);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_spatial_index_with_geometry() {
        let mut index = SpatialIndex3D::new();

        let geom1 = Geometry3D::new("B1", 10.0, 10.0, 10.0);
        let geom2 = Geometry3D::new("B2", 10.0, 10.0, 10.0);

        index.insert_geometry(0, &geom1, Vector3::new(0.0, 0.0, 0.0), 0);
        index.insert_geometry(1, &geom2, Vector3::new(20.0, 0.0, 0.0), 0);

        assert_eq!(index.len(), 2);

        // Query for potential collisions with a new geometry at (5, 0, 0)
        let query_geom = Geometry3D::new("Q", 10.0, 10.0, 10.0);
        let results = index.query_geometry(&query_geom, Vector3::new(5.0, 0.0, 0.0), 0);

        // Should intersect with first geometry only
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_has_collision() {
        let mut index = SpatialIndex3D::new();

        let geom1 = Geometry3D::new("B1", 10.0, 10.0, 10.0);
        index.insert_geometry(0, &geom1, Vector3::new(0.0, 0.0, 0.0), 0);

        let query_geom = Geometry3D::new("Q", 5.0, 5.0, 5.0);

        // Should collide (overlapping)
        assert!(index.has_collision(&query_geom, Vector3::new(5.0, 5.0, 5.0), 0, 0.0));

        // Should not collide (far away)
        assert!(!index.has_collision(&query_geom, Vector3::new(50.0, 0.0, 0.0), 0, 0.0));

        // Should collide with spacing
        assert!(index.has_collision(&query_geom, Vector3::new(12.0, 0.0, 0.0), 0, 3.0));

        // Should not collide without spacing
        assert!(!index.has_collision(&query_geom, Vector3::new(12.0, 0.0, 0.0), 0, 0.0));
    }

    #[test]
    fn test_can_place() {
        let mut index = SpatialIndex3D::new();

        let geom1 = Geometry3D::new("B1", 10.0, 10.0, 10.0);
        index.insert_geometry(0, &geom1, Vector3::new(5.0, 5.0, 5.0), 0);

        let boundary = Aabb3D::new([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]);
        let query_geom = Geometry3D::new("Q", 10.0, 10.0, 10.0);

        // Can place at (20, 20, 20) - no collision, within boundary
        assert!(index.can_place(
            &query_geom,
            Vector3::new(20.0, 20.0, 20.0),
            0,
            &boundary,
            0.0
        ));

        // Cannot place at (5, 5, 5) - collision with existing
        assert!(!index.can_place(&query_geom, Vector3::new(5.0, 5.0, 5.0), 0, &boundary, 0.0));

        // Cannot place at (95, 95, 95) - outside boundary
        assert!(!index.can_place(
            &query_geom,
            Vector3::new(95.0, 95.0, 95.0),
            0,
            &boundary,
            0.0
        ));

        // Cannot place at (-5, 5, 5) - outside boundary (negative)
        assert!(!index.can_place(&query_geom, Vector3::new(-5.0, 5.0, 5.0), 0, &boundary, 0.0));
    }

    #[test]
    fn test_transformed_aabb() {
        let geom = Geometry3D::new("B", 10.0, 20.0, 30.0);
        let aabb = compute_transformed_aabb(&geom, Vector3::new(5.0, 5.0, 5.0), 0);

        assert!((aabb.min[0] - 5.0).abs() < 1e-10);
        assert!((aabb.min[1] - 5.0).abs() < 1e-10);
        assert!((aabb.min[2] - 5.0).abs() < 1e-10);
        assert!((aabb.max[0] - 15.0).abs() < 1e-10);
        assert!((aabb.max[1] - 25.0).abs() < 1e-10);
        assert!((aabb.max[2] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_aabb() {
        let aabb = boundary_aabb(100.0, 80.0, 50.0, 5.0);

        assert!((aabb.min[0] - 5.0).abs() < 1e-10);
        assert!((aabb.min[1] - 5.0).abs() < 1e-10);
        assert!((aabb.min[2] - 5.0).abs() < 1e-10);
        assert!((aabb.max[0] - 95.0).abs() < 1e-10);
        assert!((aabb.max[1] - 75.0).abs() < 1e-10);
        assert!((aabb.max[2] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_potential_collisions() {
        let mut index = SpatialIndex3D::new();

        let geom1 = Geometry3D::new("B1", 10.0, 10.0, 10.0);
        let geom2 = Geometry3D::new("B2", 10.0, 10.0, 10.0);
        let geom3 = Geometry3D::new("B3", 10.0, 10.0, 10.0);

        index.insert_geometry(0, &geom1, Vector3::new(0.0, 0.0, 0.0), 0);
        index.insert_geometry(1, &geom2, Vector3::new(50.0, 0.0, 0.0), 0);
        index.insert_geometry(2, &geom3, Vector3::new(0.0, 50.0, 0.0), 0);

        let query_geom = Geometry3D::new("Q", 5.0, 5.0, 5.0);

        // Query at (5, 5, 5) with spacing 2 should only collide with index 0
        let collisions =
            index.get_potential_collisions(&query_geom, Vector3::new(5.0, 5.0, 5.0), 0, 2.0);
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0], 0);
    }
}
