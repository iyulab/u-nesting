//! Spatial indexing for 2D collision detection using R*-tree.
//!
//! This module provides efficient broad-phase collision detection for the nesting
//! algorithm by using an R*-tree spatial index to quickly identify potentially
//! overlapping geometries.

use crate::geometry::Geometry2D;
use rstar::{primitives::Rectangle, RTree, RTreeObject, AABB};
use u_nesting_core::geometry::Geometry;

/// An entry in the 2D spatial index representing a placed geometry.
#[derive(Debug, Clone)]
pub struct SpatialEntry2D {
    /// Index of the geometry in the placed list
    pub index: usize,
    /// Geometry ID
    pub id: String,
    /// Axis-aligned bounding box (min_x, min_y, max_x, max_y)
    pub aabb: [f64; 4],
}

impl SpatialEntry2D {
    /// Creates a new spatial entry from a placed geometry.
    pub fn new(index: usize, id: String, aabb: [f64; 4]) -> Self {
        Self { index, id, aabb }
    }

    /// Creates a spatial entry from a geometry at a given position with rotation.
    pub fn from_placed(
        index: usize,
        geometry: &Geometry2D,
        position: (f64, f64),
        rotation: f64,
    ) -> Self {
        let aabb = compute_transformed_aabb(geometry, position, rotation);
        Self {
            index,
            id: geometry.id().clone(),
            aabb,
        }
    }
}

impl RTreeObject for SpatialEntry2D {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.aabb[0], self.aabb[1]], [self.aabb[2], self.aabb[3]])
    }
}

/// 2D spatial index using R*-tree for efficient collision queries.
#[derive(Debug)]
pub struct SpatialIndex2D {
    tree: RTree<SpatialEntry2D>,
}

impl SpatialIndex2D {
    /// Creates a new empty spatial index.
    pub fn new() -> Self {
        Self { tree: RTree::new() }
    }

    /// Creates a spatial index with the given entries.
    pub fn with_entries(entries: Vec<SpatialEntry2D>) -> Self {
        Self {
            tree: RTree::bulk_load(entries),
        }
    }

    /// Inserts a new entry into the spatial index.
    pub fn insert(&mut self, entry: SpatialEntry2D) {
        self.tree.insert(entry);
    }

    /// Inserts a placed geometry into the spatial index.
    pub fn insert_geometry(
        &mut self,
        index: usize,
        geometry: &Geometry2D,
        position: (f64, f64),
        rotation: f64,
    ) {
        let entry = SpatialEntry2D::from_placed(index, geometry, position, rotation);
        self.insert(entry);
    }

    /// Returns the number of entries in the index.
    pub fn len(&self) -> usize {
        self.tree.size()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.tree.size() == 0
    }

    /// Clears all entries from the index.
    pub fn clear(&mut self) {
        self.tree = RTree::new();
    }

    /// Finds all entries whose bounding boxes intersect with the given AABB.
    ///
    /// This is the primary broad-phase collision detection method.
    pub fn query_aabb(&self, min: [f64; 2], max: [f64; 2]) -> Vec<&SpatialEntry2D> {
        let envelope = AABB::from_corners(min, max);
        self.tree
            .locate_in_envelope_intersecting(&envelope)
            .collect()
    }

    /// Finds all entries that potentially intersect with a geometry at the given position.
    pub fn query_geometry(
        &self,
        geometry: &Geometry2D,
        position: (f64, f64),
        rotation: f64,
    ) -> Vec<&SpatialEntry2D> {
        let aabb = compute_transformed_aabb(geometry, position, rotation);
        self.query_aabb([aabb[0], aabb[1]], [aabb[2], aabb[3]])
    }

    /// Finds all entries that potentially intersect with the given geometry AABB
    /// expanded by a margin (for spacing).
    pub fn query_with_margin(
        &self,
        geometry: &Geometry2D,
        position: (f64, f64),
        rotation: f64,
        margin: f64,
    ) -> Vec<&SpatialEntry2D> {
        let aabb = compute_transformed_aabb(geometry, position, rotation);
        self.query_aabb(
            [aabb[0] - margin, aabb[1] - margin],
            [aabb[2] + margin, aabb[3] + margin],
        )
    }

    /// Returns an iterator over all entries in the index.
    pub fn iter(&self) -> impl Iterator<Item = &SpatialEntry2D> {
        self.tree.iter()
    }

    /// Returns the indices of potentially colliding geometries for a query geometry.
    pub fn get_potential_collisions(
        &self,
        geometry: &Geometry2D,
        position: (f64, f64),
        rotation: f64,
        spacing: f64,
    ) -> Vec<usize> {
        self.query_with_margin(geometry, position, rotation, spacing)
            .iter()
            .map(|entry| entry.index)
            .collect()
    }
}

impl Default for SpatialIndex2D {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the transformed AABB of a geometry at a given position with rotation.
pub fn compute_transformed_aabb(
    geometry: &Geometry2D,
    position: (f64, f64),
    rotation: f64,
) -> [f64; 4] {
    if rotation.abs() < 1e-10 {
        // No rotation - simple translation
        let (g_min, g_max) = geometry.aabb();
        [
            g_min[0] + position.0,
            g_min[1] + position.1,
            g_max[0] + position.0,
            g_max[1] + position.1,
        ]
    } else {
        // Apply rotation then translation
        let exterior = geometry.exterior();
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for (x, y) in exterior {
            // Rotate around origin
            let rx = x * cos_r - y * sin_r;
            let ry = x * sin_r + y * cos_r;
            // Translate
            let tx = rx + position.0;
            let ty = ry + position.1;

            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }

        [min_x, min_y, max_x, max_y]
    }
}

/// A wrapper around Rectangle for R*-tree that stores additional metadata.
#[derive(Debug, Clone)]
pub struct IndexedRectangle {
    rectangle: Rectangle<[f64; 2]>,
    pub index: usize,
}

impl IndexedRectangle {
    /// Creates a new indexed rectangle.
    pub fn new(min: [f64; 2], max: [f64; 2], index: usize) -> Self {
        Self {
            rectangle: Rectangle::from_corners(min, max),
            index,
        }
    }
}

impl RTreeObject for IndexedRectangle {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.rectangle.envelope()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_index_new() {
        let index = SpatialIndex2D::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_index_insert() {
        let mut index = SpatialIndex2D::new();
        let entry = SpatialEntry2D::new(0, "test".to_string(), [0.0, 0.0, 10.0, 10.0]);
        index.insert(entry);

        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_spatial_index_query_aabb() {
        let mut index = SpatialIndex2D::new();

        // Insert three non-overlapping rectangles
        index.insert(SpatialEntry2D::new(
            0,
            "r1".to_string(),
            [0.0, 0.0, 10.0, 10.0],
        ));
        index.insert(SpatialEntry2D::new(
            1,
            "r2".to_string(),
            [20.0, 0.0, 30.0, 10.0],
        ));
        index.insert(SpatialEntry2D::new(
            2,
            "r3".to_string(),
            [0.0, 20.0, 10.0, 30.0],
        ));

        // Query overlapping with r1 only
        let results = index.query_aabb([5.0, 5.0], [15.0, 15.0]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);

        // Query overlapping with r1 and r2
        let results = index.query_aabb([5.0, 0.0], [25.0, 10.0]);
        assert_eq!(results.len(), 2);

        // Query overlapping with nothing
        let results = index.query_aabb([50.0, 50.0], [60.0, 60.0]);
        assert!(results.is_empty());

        // Query overlapping with all
        let results = index.query_aabb([-10.0, -10.0], [40.0, 40.0]);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_spatial_index_with_geometry() {
        let mut index = SpatialIndex2D::new();

        let geom1 = Geometry2D::rectangle("R1", 10.0, 10.0);
        let geom2 = Geometry2D::rectangle("R2", 10.0, 10.0);

        index.insert_geometry(0, &geom1, (0.0, 0.0), 0.0);
        index.insert_geometry(1, &geom2, (20.0, 0.0), 0.0);

        assert_eq!(index.len(), 2);

        // Query for potential collisions with a new geometry at (5, 0)
        let query_geom = Geometry2D::rectangle("Q", 10.0, 10.0);
        let results = index.query_geometry(&query_geom, (5.0, 0.0), 0.0);

        // Should intersect with first geometry only
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_transformed_aabb_no_rotation() {
        let geom = Geometry2D::rectangle("R", 10.0, 10.0);
        let aabb = compute_transformed_aabb(&geom, (5.0, 5.0), 0.0);

        assert!((aabb[0] - 5.0).abs() < 1e-10);
        assert!((aabb[1] - 5.0).abs() < 1e-10);
        assert!((aabb[2] - 15.0).abs() < 1e-10);
        assert!((aabb[3] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_transformed_aabb_with_rotation() {
        let geom = Geometry2D::rectangle("R", 10.0, 10.0);
        let rotation = std::f64::consts::FRAC_PI_4; // 45 degrees
        let aabb = compute_transformed_aabb(&geom, (0.0, 0.0), rotation);

        // 10x10 square at origin, rotated 45 degrees:
        // Original vertices: (0,0), (10,0), (10,10), (0,10)
        // After rotation:
        // (0, 0) -> (0, 0)
        // (10, 0) -> (7.07, 7.07)
        // (10, 10) -> (0, 14.14)
        // (0, 10) -> (-7.07, 7.07)
        // Resulting AABB: min_x ~ -7.07, min_y ~ 0, max_x ~ 7.07, max_y ~ 14.14
        let half_diag = 10.0 * std::f64::consts::FRAC_1_SQRT_2;
        assert!(aabb[0] < -half_diag + 0.1); // min_x should be around -7.07
        assert!(aabb[1].abs() < 0.1); // min_y should be close to 0
        assert!(aabb[2] > half_diag - 0.1); // max_x should be around 7.07
        assert!((aabb[3] - 10.0 * std::f64::consts::SQRT_2).abs() < 0.1); // max_y ~ 14.14
    }

    #[test]
    fn test_query_with_margin() {
        let mut index = SpatialIndex2D::new();

        // Two rectangles with a gap
        index.insert(SpatialEntry2D::new(
            0,
            "r1".to_string(),
            [0.0, 0.0, 10.0, 10.0],
        ));
        index.insert(SpatialEntry2D::new(
            1,
            "r2".to_string(),
            [15.0, 0.0, 25.0, 10.0],
        ));

        let query_geom = Geometry2D::rectangle("Q", 5.0, 5.0);

        // Without margin, positioned at (10.5, 0) should not intersect
        let results = index.query_geometry(&query_geom, (10.5, 0.0), 0.0);
        // Actually at (10.5, 0) with width 5, AABB is [10.5, 0, 15.5, 5] - intersects r2
        assert_eq!(results.len(), 1);

        // With margin of 3.0, should intersect both
        let results = index.query_with_margin(&query_geom, (10.5, 0.0), 0.0, 3.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_potential_collisions() {
        let mut index = SpatialIndex2D::new();

        let geom1 = Geometry2D::rectangle("R1", 10.0, 10.0);
        let geom2 = Geometry2D::rectangle("R2", 10.0, 10.0);
        let geom3 = Geometry2D::rectangle("R3", 10.0, 10.0);

        index.insert_geometry(0, &geom1, (0.0, 0.0), 0.0);
        index.insert_geometry(1, &geom2, (50.0, 0.0), 0.0);
        index.insert_geometry(2, &geom3, (0.0, 50.0), 0.0);

        let query_geom = Geometry2D::rectangle("Q", 5.0, 5.0);

        // Query at (5, 5) with spacing 2 should only collide with index 0
        let collisions = index.get_potential_collisions(&query_geom, (5.0, 5.0), 0.0, 2.0);
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0], 0);
    }

    #[test]
    fn test_bulk_load() {
        let entries = vec![
            SpatialEntry2D::new(0, "r1".to_string(), [0.0, 0.0, 10.0, 10.0]),
            SpatialEntry2D::new(1, "r2".to_string(), [20.0, 0.0, 30.0, 10.0]),
            SpatialEntry2D::new(2, "r3".to_string(), [0.0, 20.0, 10.0, 30.0]),
        ];

        let index = SpatialIndex2D::with_entries(entries);
        assert_eq!(index.len(), 3);

        let results = index.query_aabb([0.0, 0.0], [15.0, 15.0]);
        assert_eq!(results.len(), 1);
    }
}
