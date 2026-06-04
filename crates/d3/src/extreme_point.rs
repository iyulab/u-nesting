//! Extreme Point heuristic for 3D bin packing.
//!
//! This module implements the Extreme Point (EP) heuristic for 3D bin packing,
//! which is more efficient than simple layer-based packing for many real-world
//! scenarios.
//!
//! # Algorithm Overview
//!
//! Extreme Points are positions where a new box could be placed touching at least
//! two surfaces (walls or other boxes). When a box is placed, it generates new
//! extreme points at its corners and edges.
//!
//! # References
//!
//! - Crainic, T. G., Perboli, G., & Tadei, R. (2008). Extreme point-based heuristics
//!   for three-dimensional bin packing.

use crate::boundary::Boundary3D;
use crate::geometry::Geometry3D;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use u_nesting_core::geom::nalgebra_types::NaVector3 as Vector3;
use u_nesting_core::geometry::{Boundary, Geometry};

/// A 3D point representing a potential placement position.
#[derive(Debug, Clone, Copy)]
pub struct ExtremePoint {
    /// Position (x, y, z).
    pub position: Vector3<f64>,
    /// The residual space in x direction from this point.
    pub residual_x: f64,
    /// The residual space in y direction from this point.
    pub residual_y: f64,
    /// The residual space in z direction from this point.
    pub residual_z: f64,
}

impl ExtremePoint {
    /// Creates a new extreme point.
    pub fn new(x: f64, y: f64, z: f64, res_x: f64, res_y: f64, res_z: f64) -> Self {
        Self {
            position: Vector3::new(x, y, z),
            residual_x: res_x,
            residual_y: res_y,
            residual_z: res_z,
        }
    }

    /// Returns the position as a tuple.
    pub fn pos(&self) -> (f64, f64, f64) {
        (self.position.x, self.position.y, self.position.z)
    }

    /// Checks if a box with given dimensions fits at this point.
    pub fn fits(&self, width: f64, depth: f64, height: f64) -> bool {
        width <= self.residual_x + 1e-9
            && depth <= self.residual_y + 1e-9
            && height <= self.residual_z + 1e-9
    }
}

impl PartialEq for ExtremePoint {
    fn eq(&self, other: &Self) -> bool {
        (self.position - other.position).norm() < 1e-9
    }
}

impl Eq for ExtremePoint {}

/// Wrapper for BinaryHeap ordering (min-heap by z, then y, then x).
#[derive(Debug, Clone)]
struct OrderedEP(ExtremePoint);

impl PartialEq for OrderedEP {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedEP {}

impl PartialOrd for OrderedEP {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedEP {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: prefer lower z, then lower y, then lower x (reversed for BinaryHeap)
        let z_cmp = other
            .0
            .position
            .z
            .partial_cmp(&self.0.position.z)
            .unwrap_or(Ordering::Equal);
        if z_cmp != Ordering::Equal {
            return z_cmp;
        }

        let y_cmp = other
            .0
            .position
            .y
            .partial_cmp(&self.0.position.y)
            .unwrap_or(Ordering::Equal);
        if y_cmp != Ordering::Equal {
            return y_cmp;
        }

        other
            .0
            .position
            .x
            .partial_cmp(&self.0.position.x)
            .unwrap_or(Ordering::Equal)
    }
}

/// A placed box in the container.
#[derive(Debug, Clone)]
pub struct PlacedBox {
    /// Geometry ID.
    pub id: String,
    /// Instance number.
    pub instance: usize,
    /// Position (min corner).
    pub position: Vector3<f64>,
    /// Dimensions after orientation applied.
    pub dimensions: Vector3<f64>,
    /// Mass of the box.
    pub mass: Option<f64>,
}

impl PlacedBox {
    /// Returns the max corner of the box.
    pub fn max_corner(&self) -> Vector3<f64> {
        self.position + self.dimensions
    }

    /// Checks if this box overlaps with another box.
    pub fn overlaps(&self, other: &PlacedBox) -> bool {
        let self_max = self.max_corner();
        let other_max = other.max_corner();

        // Check for non-overlap in each dimension
        let no_overlap_x =
            self.position.x >= other_max.x - 1e-9 || other.position.x >= self_max.x - 1e-9;
        let no_overlap_y =
            self.position.y >= other_max.y - 1e-9 || other.position.y >= self_max.y - 1e-9;
        let no_overlap_z =
            self.position.z >= other_max.z - 1e-9 || other.position.z >= self_max.z - 1e-9;

        !(no_overlap_x || no_overlap_y || no_overlap_z)
    }
}

/// Extreme Point Set manager.
pub struct ExtremePointSet {
    /// Priority queue of extreme points (min-heap by z, y, x).
    points: BinaryHeap<OrderedEP>,
    /// Container dimensions.
    container: Vector3<f64>,
    /// Placed boxes.
    placed: Vec<PlacedBox>,
    /// Spacing between boxes.
    spacing: f64,
    /// Margin from container walls.
    margin: f64,
}

impl ExtremePointSet {
    /// Creates a new extreme point set for a container.
    pub fn new(boundary: &Boundary3D, margin: f64, spacing: f64) -> Self {
        let container = Vector3::new(boundary.width(), boundary.depth(), boundary.height());

        let mut eps = Self {
            points: BinaryHeap::new(),
            container,
            placed: Vec::new(),
            spacing,
            margin,
        };

        // Initial extreme point at origin (with margin)
        let initial_ep = ExtremePoint::new(
            margin,
            margin,
            margin,
            container.x - 2.0 * margin,
            container.y - 2.0 * margin,
            container.z - 2.0 * margin,
        );
        eps.points.push(OrderedEP(initial_ep));

        eps
    }

    /// Returns the number of extreme points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the number of placed boxes.
    pub fn placed_count(&self) -> usize {
        self.placed.len()
    }

    /// Returns total placed volume.
    pub fn total_volume(&self) -> f64 {
        self.placed
            .iter()
            .map(|b| b.dimensions.x * b.dimensions.y * b.dimensions.z)
            .sum()
    }

    /// Returns total placed mass.
    pub fn total_mass(&self) -> f64 {
        self.placed.iter().filter_map(|b| b.mass).sum()
    }

    /// Returns the placed boxes.
    pub fn placed_boxes(&self) -> &[PlacedBox] {
        &self.placed
    }

    /// Tries to place a box, returns the placement position if successful.
    pub fn try_place(
        &mut self,
        geom: &Geometry3D,
        instance: usize,
        orientation: usize,
    ) -> Option<Vector3<f64>> {
        let dims = geom.dimensions_for_orientation(orientation);
        let width = dims.x + self.spacing;
        let depth = dims.y + self.spacing;
        let height = dims.z + self.spacing;

        // Collect all current EPs
        let mut candidates: Vec<ExtremePoint> = Vec::new();
        while let Some(OrderedEP(ep)) = self.points.pop() {
            candidates.push(ep);
        }

        // Find the best fitting EP
        let mut best_ep_idx: Option<usize> = None;
        for (idx, ep) in candidates.iter().enumerate() {
            // The orientation (and therefore dims) is already fixed by the caller's
            // loop, so the box must fit at this EP with its actual extents. A prior
            // version also tested `fits(width, height, depth)` — swapping depth/height
            // — which let a box pass on residuals it did not actually occupy and then
            // get placed with its real dims, producing out-of-bounds placements.
            if ep.fits(width, depth, height) {
                // Check if placement would overlap with existing boxes
                let test_box = PlacedBox {
                    id: String::new(),
                    instance: 0,
                    position: ep.position,
                    dimensions: dims,
                    mass: None,
                };

                let overlaps = self.placed.iter().any(|placed| test_box.overlaps(placed));
                if !overlaps {
                    best_ep_idx = Some(idx);
                    break;
                }
            }
        }

        // Restore non-used EPs
        let result = if let Some(idx) = best_ep_idx {
            let chosen_ep = candidates.remove(idx);

            // Place the box
            let placed_box = PlacedBox {
                id: geom.id().clone(),
                instance,
                position: chosen_ep.position,
                dimensions: dims,
                mass: geom.mass(),
            };

            // Generate new extreme points
            self.generate_new_eps(&placed_box);

            let position = chosen_ep.position;
            self.placed.push(placed_box);

            Some(position)
        } else {
            None
        };

        // Return remaining candidates to the heap
        for ep in candidates {
            self.points.push(OrderedEP(ep));
        }

        result
    }

    /// Generates new extreme points after placing a box.
    fn generate_new_eps(&mut self, placed: &PlacedBox) {
        let box_max = placed.max_corner();
        let container_max = self.container - Vector3::new(self.margin, self.margin, self.margin);

        // EP1: Top-right-front of the box (x direction)
        if box_max.x < container_max.x {
            let res_x = container_max.x - box_max.x;
            let res_y = self.compute_residual_y(box_max.x, placed.position.y, placed.position.z);
            let res_z = self.compute_residual_z(box_max.x, placed.position.y, placed.position.z);

            if res_x > 1e-9 && res_y > 1e-9 && res_z > 1e-9 {
                let ep = ExtremePoint::new(
                    box_max.x,
                    placed.position.y,
                    placed.position.z,
                    res_x,
                    res_y,
                    res_z,
                );
                self.add_ep_if_valid(ep);
            }
        }

        // EP2: Top-right-front of the box (y direction)
        if box_max.y < container_max.y {
            let res_x = self.compute_residual_x(placed.position.x, box_max.y, placed.position.z);
            let res_y = container_max.y - box_max.y;
            let res_z = self.compute_residual_z(placed.position.x, box_max.y, placed.position.z);

            if res_x > 1e-9 && res_y > 1e-9 && res_z > 1e-9 {
                let ep = ExtremePoint::new(
                    placed.position.x,
                    box_max.y,
                    placed.position.z,
                    res_x,
                    res_y,
                    res_z,
                );
                self.add_ep_if_valid(ep);
            }
        }

        // EP3: Top of the box (z direction)
        if box_max.z < container_max.z {
            let res_x = self.compute_residual_x(placed.position.x, placed.position.y, box_max.z);
            let res_y = self.compute_residual_y(placed.position.x, placed.position.y, box_max.z);
            let res_z = container_max.z - box_max.z;

            if res_x > 1e-9 && res_y > 1e-9 && res_z > 1e-9 {
                let ep = ExtremePoint::new(
                    placed.position.x,
                    placed.position.y,
                    box_max.z,
                    res_x,
                    res_y,
                    res_z,
                );
                self.add_ep_if_valid(ep);
            }
        }
    }

    /// Adds an EP if it's valid and not dominated by existing EPs.
    fn add_ep_if_valid(&mut self, ep: ExtremePoint) {
        // Check bounds
        let container_max = self.container - Vector3::new(self.margin, self.margin, self.margin);
        if ep.position.x >= container_max.x - 1e-9
            || ep.position.y >= container_max.y - 1e-9
            || ep.position.z >= container_max.z - 1e-9
        {
            return;
        }

        // Check if position is inside any placed box
        for placed in &self.placed {
            let max = placed.max_corner();
            if ep.position.x > placed.position.x - 1e-9
                && ep.position.x < max.x + 1e-9
                && ep.position.y > placed.position.y - 1e-9
                && ep.position.y < max.y + 1e-9
                && ep.position.z > placed.position.z - 1e-9
                && ep.position.z < max.z + 1e-9
            {
                return;
            }
        }

        self.points.push(OrderedEP(ep));
    }

    /// Computes residual space in x direction from a given point.
    fn compute_residual_x(&self, x: f64, y: f64, z: f64) -> f64 {
        let container_max_x = self.container.x - self.margin;
        let mut min_x = container_max_x;

        for placed in &self.placed {
            let p_max = placed.max_corner();
            // Check if this box blocks in x direction
            if placed.position.y < y + 1e-9
                && p_max.y > y - 1e-9
                && placed.position.z < z + 1e-9
                && p_max.z > z - 1e-9
                && placed.position.x > x - 1e-9
                && placed.position.x < min_x
            {
                min_x = placed.position.x;
            }
        }

        (min_x - x).max(0.0)
    }

    /// Computes residual space in y direction from a given point.
    fn compute_residual_y(&self, x: f64, y: f64, z: f64) -> f64 {
        let container_max_y = self.container.y - self.margin;
        let mut min_y = container_max_y;

        for placed in &self.placed {
            let p_max = placed.max_corner();
            // Check if this box blocks in y direction
            if placed.position.x < x + 1e-9
                && p_max.x > x - 1e-9
                && placed.position.z < z + 1e-9
                && p_max.z > z - 1e-9
                && placed.position.y > y - 1e-9
                && placed.position.y < min_y
            {
                min_y = placed.position.y;
            }
        }

        (min_y - y).max(0.0)
    }

    /// Computes residual space in z direction from a given point.
    fn compute_residual_z(&self, x: f64, y: f64, z: f64) -> f64 {
        let container_max_z = self.container.z - self.margin;
        let mut min_z = container_max_z;

        for placed in &self.placed {
            let p_max = placed.max_corner();
            // Check if this box blocks in z direction
            if placed.position.x < x + 1e-9
                && p_max.x > x - 1e-9
                && placed.position.y < y + 1e-9
                && p_max.y > y - 1e-9
                && placed.position.z > z - 1e-9
                && placed.position.z < min_z
            {
                min_z = placed.position.z;
            }
        }

        (min_z - z).max(0.0)
    }
}

/// EP selection strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum EpSelectionStrategy {
    /// Select EP with lowest z, then y, then x (default).
    #[default]
    BottomLeftBack,
    /// Select EP that minimizes wasted space.
    BestFit,
    /// Select first fitting EP.
    FirstFit,
}

/// Result of EP packing: (geometry_id, instance, position, orientation).
pub type EpPlacement = (String, usize, Vector3<f64>, usize);

/// Runs EP-based packing.
pub fn run_ep_packing(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    margin: f64,
    spacing: f64,
    max_mass: Option<f64>,
) -> (Vec<EpPlacement>, f64) {
    let mut eps = ExtremePointSet::new(boundary, margin, spacing);
    let mut placements = Vec::new();

    // Sort geometries by volume (largest first) for better packing
    let mut items: Vec<(usize, usize, f64)> = Vec::new();
    for (geom_idx, geom) in geometries.iter().enumerate() {
        for instance in 0..geom.quantity() {
            items.push((geom_idx, instance, geom.measure()));
        }
    }
    items.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    for (geom_idx, instance, _) in items {
        let geom = &geometries[geom_idx];

        // Check mass constraint
        if let (Some(max), Some(item_mass)) = (max_mass, geom.mass()) {
            if eps.total_mass() + item_mass > max {
                continue;
            }
        }

        // Try each orientation
        let mut placed = false;
        for orientation in 0..geom.allowed_orientations().len() {
            if let Some(position) = eps.try_place(geom, instance, orientation) {
                placements.push((geom.id().clone(), instance, position, orientation));
                placed = true;
                break;
            }
        }

        if !placed {
            // Could not place this item
        }
    }

    let utilization = eps.total_volume() / boundary.measure();
    (placements, utilization)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extreme_point_creation() {
        let ep = ExtremePoint::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
        assert!(ep.fits(50.0, 50.0, 50.0));
        assert!(!ep.fits(150.0, 50.0, 50.0));
    }

    #[test]
    fn test_extreme_point_set_initial() {
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let eps = ExtremePointSet::new(&boundary, 0.0, 0.0);

        assert_eq!(eps.len(), 1);
        assert!(!eps.is_empty());
    }

    #[test]
    fn test_ep_packing_single_box() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);

        let (placements, utilization) = run_ep_packing(&geometries, &boundary, 0.0, 0.0, None);

        assert_eq!(placements.len(), 1);
        assert!(utilization > 0.0);
    }

    #[test]
    fn test_ep_packing_multiple_boxes() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(8)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);

        let (placements, utilization) = run_ep_packing(&geometries, &boundary, 0.0, 0.0, None);

        // Should be able to fit multiple boxes
        assert!(placements.len() >= 4);
        assert!(utilization > 0.05);
    }

    #[test]
    fn test_ep_packing_all_placements_in_bounds() {
        // Regression for the 0.3.3 EP out-of-bounds bug: `try_place` accepted an EP
        // via `fits(width, height, depth)` (depth/height swapped), then placed the box
        // with its real dims, overflowing the boundary. These configs each produced
        // out-of-bounds placements before the fix (y- and z-violations); every
        // placement must now lie fully inside the boundary for its chosen orientation.
        use crate::geometry::OrientationConstraint::Any;
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let configs = [
            ("flat-deep", 60.0, 70.0, 15.0, 4usize), // pre-fix: pos(0,70,0) -> y=140
            ("tall-thin", 40.0, 10.0, 40.0, 20usize), // pre-fix: pos(0,0,80) -> z=120
            ("issue-mid", 30.0, 20.0, 25.0, 6usize),
        ];

        for (label, w, d, h, qty) in configs {
            let geometries = vec![Geometry3D::new(label, w, d, h)
                .with_quantity(qty)
                .with_orientation(Any)];
            let (placements, _) = run_ep_packing(&geometries, &boundary, 0.0, 0.0, None);

            for (id, instance, pos, orientation) in &placements {
                let dims = geometries[0].dimensions_for_orientation(*orientation);
                let eps = 1e-6;
                assert!(
                    pos.x + dims.x <= boundary.width() + eps
                        && pos.y + dims.y <= boundary.depth() + eps
                        && pos.z + dims.z <= boundary.height() + eps
                        && pos.x >= -eps
                        && pos.y >= -eps
                        && pos.z >= -eps,
                    "[{label}] {id}#{instance} out of bounds: pos({:.1},{:.1},{:.1}) \
                     dims({:.1},{:.1},{:.1}) orientation {orientation} exceeds 100x100x100",
                    pos.x,
                    pos.y,
                    pos.z,
                    dims.x,
                    dims.y,
                    dims.z,
                );
            }
        }
    }

    #[test]
    fn test_ep_packing_with_margin() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(4)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);

        let (placements, _) = run_ep_packing(&geometries, &boundary, 5.0, 0.0, None);

        // With margin, first box should start at (5, 5, 5)
        if !placements.is_empty() {
            let (_, _, pos, _) = &placements[0];
            assert!(pos.x >= 4.9);
            assert!(pos.y >= 4.9);
            assert!(pos.z >= 4.9);
        }
    }

    #[test]
    fn test_ep_packing_with_spacing() {
        let geometries = vec![Geometry3D::new("B1", 40.0, 40.0, 40.0).with_quantity(4)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);

        let (placements_no_spacing, _) = run_ep_packing(&geometries, &boundary, 0.0, 0.0, None);
        let (placements_with_spacing, _) = run_ep_packing(&geometries, &boundary, 0.0, 5.0, None);

        // With spacing, fewer boxes might fit
        assert!(placements_with_spacing.len() <= placements_no_spacing.len());
    }

    #[test]
    fn test_placed_box_overlap() {
        let box1 = PlacedBox {
            id: "A".to_string(),
            instance: 0,
            position: Vector3::new(0.0, 0.0, 0.0),
            dimensions: Vector3::new(10.0, 10.0, 10.0),
            mass: None,
        };

        let box2_overlap = PlacedBox {
            id: "B".to_string(),
            instance: 0,
            position: Vector3::new(5.0, 5.0, 5.0),
            dimensions: Vector3::new(10.0, 10.0, 10.0),
            mass: None,
        };

        let box2_no_overlap = PlacedBox {
            id: "C".to_string(),
            instance: 0,
            position: Vector3::new(15.0, 0.0, 0.0),
            dimensions: Vector3::new(10.0, 10.0, 10.0),
            mass: None,
        };

        assert!(box1.overlaps(&box2_overlap));
        assert!(!box1.overlaps(&box2_no_overlap));
    }

    #[test]
    fn test_ep_packing_orientations() {
        use crate::geometry::OrientationConstraint;

        // Long box that benefits from rotation
        let geometries = vec![Geometry3D::new("B1", 80.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);

        let (placements, _) = run_ep_packing(&geometries, &boundary, 0.0, 0.0, None);

        // With orientation flexibility, both should fit
        assert_eq!(placements.len(), 2);
    }
}
