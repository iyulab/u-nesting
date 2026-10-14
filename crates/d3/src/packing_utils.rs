//! Shared utilities for 3D bin packing solvers.
//!
//! This module consolidates common code across GA, SA, and BRKGA packing solvers,
//! following the same deduplication pattern as d2's `placement_utils`.
//!
//! # Extracted Components
//!
//! - [`InstanceInfo`]: Maps expanded instances back to their source geometry
//! - [`packing_fitness`]: Unified fitness formula for all solvers
//! - [`build_unplaced_list`]: Identifies geometries not placed in the solution
//! - [`layer_place_items`]: Core layer-based placement algorithm shared by all solvers

use crate::boundary::Boundary3D;
use crate::geometry::Geometry3D;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::solver::Config;
use u_nesting_core::Placement;

/// Instance information mapping expanded instances to source geometries.
///
/// When a geometry has quantity > 1, it expands into multiple instances.
/// This struct tracks which geometry each instance belongs to and its
/// ordinal within that geometry's quantity.
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// Index into the geometries array.
    pub geometry_idx: usize,
    /// Instance number within this geometry's quantity.
    pub instance_num: usize,
    /// Number of allowed orientations.
    pub orientation_count: usize,
}

/// Builds the instance mapping from geometries.
///
/// Expands each geometry by its quantity, recording the geometry index,
/// instance number, and orientation count for each expanded instance.
pub fn build_instances(geometries: &[Geometry3D]) -> Vec<InstanceInfo> {
    let mut instances = Vec::new();
    for (geom_idx, geom) in geometries.iter().enumerate() {
        let orient_count = geom.allowed_orientations().len();
        for instance_num in 0..geom.quantity() {
            instances.push(InstanceInfo {
                geometry_idx: geom_idx,
                instance_num,
                orientation_count: orient_count,
            });
        }
    }
    instances
}

/// Represents a single item to be placed, with its resolved orientation.
pub struct PlacementItem {
    /// Index into the instances array.
    pub instance_idx: usize,
    /// Resolved orientation index for this item.
    pub orientation_idx: usize,
}

/// Computes packing fitness using the unified formula.
///
/// The fitness combines placement ratio and volume utilization:
/// `fitness = (placed / total) * 100 + utilization * 10`
///
/// This prioritizes placing more items (100x weight) while also
/// rewarding tighter packing (10x weight).
///
/// # Arguments
/// * `placed_count` - Number of items successfully placed
/// * `total_count` - Total number of items to place
/// * `utilization` - Volume utilization ratio (0.0 to 1.0)
pub fn packing_fitness(placed_count: usize, total_count: usize, utilization: f64) -> f64 {
    let placement_ratio = placed_count as f64 / total_count.max(1) as f64;
    placement_ratio * 100.0 + utilization * 10.0
}

/// The geometry ids with a copy `placements` does not contain, one entry per
/// missing copy -- [`u_nesting_core::result::unplaced_ids`] for boxes.
pub fn build_unplaced_list(
    placements: &[Placement<f64>],
    geometries: &[Geometry3D],
) -> Vec<String> {
    u_nesting_core::result::unplaced_ids(placements, geometries)
}

/// Result of the layer-based placement algorithm.
pub struct LayerPlacementResult {
    /// Successfully placed items.
    pub placements: Vec<Placement<f64>>,
    /// Volume utilization ratio.
    pub utilization: f64,
    /// Number of items placed.
    pub placed_count: usize,
}

/// Where the layer fill stands: the next spot in the current row, the depth
/// of that row and the height of the current layer.
///
/// Boxes go left to right in rows (x), rows front to back in layers (y),
/// layers bottom to top (z). A box that does not fit where the row stands
/// opens a new row, then a new layer; one that does not fit there either is
/// skipped and leaves the cursor where it was.
#[derive(Debug, Clone, Copy)]
struct LayerCursor {
    x: f64,
    y: f64,
    z: f64,
    row_depth: f64,
    layer_height: f64,
}

/// A spot a box can take, and the cursor after it is placed there.
#[derive(Debug, Clone, Copy)]
struct LayerSpot {
    x: f64,
    y: f64,
    z: f64,
    next: LayerCursor,
}

/// Layer-by-layer placement into one container: the one placement rule the
/// layer fill ([`layer_fill_in_order`]) and the search decoders
/// ([`layer_place_items`]) share, so a layout the fill produced is the layout
/// its genes decode to.
pub struct LayerFill<'a> {
    boundary: &'a Boundary3D,
    margin: f64,
    spacing: f64,
    max: [f64; 3],
    cursor: LayerCursor,
    placements: Vec<Placement<f64>>,
    volume: f64,
    mass: f64,
}

impl<'a> LayerFill<'a> {
    /// An empty container, its usable space inset by `config.margin`.
    pub fn new(boundary: &'a Boundary3D, config: &Config) -> Self {
        let margin = config.margin;
        Self {
            boundary,
            margin,
            spacing: config.spacing,
            max: [
                boundary.width() - margin,
                boundary.depth() - margin,
                boundary.height() - margin,
            ],
            cursor: LayerCursor {
                x: margin,
                y: margin,
                z: margin,
                row_depth: 0.0,
                layer_height: 0.0,
            },
            placements: Vec::new(),
            volume: 0.0,
            mass: 0.0,
        }
    }

    /// Where a box of `dims` would go next, if anywhere.
    fn spot(&self, dims: [f64; 3]) -> Option<LayerSpot> {
        let [w, d, h] = dims;
        let mut c = self.cursor;
        if c.x + w > self.max[0] {
            c.x = self.margin;
            c.y += c.row_depth + self.spacing;
            c.row_depth = 0.0;
        }
        if c.y + d > self.max[1] {
            c.x = self.margin;
            c.y = self.margin;
            c.z += c.layer_height + self.spacing;
            c.row_depth = 0.0;
            c.layer_height = 0.0;
        }
        // A fresh row or layer starts at the margin, so a box wider or deeper
        // than the container never fits there either.
        if c.x + w > self.max[0] || c.y + d > self.max[1] || c.z + h > self.max[2] {
            return None;
        }
        Some(LayerSpot {
            x: c.x,
            y: c.y,
            z: c.z,
            next: LayerCursor {
                x: c.x + w + self.spacing,
                y: c.y,
                z: c.z,
                row_depth: c.row_depth.max(d),
                layer_height: c.layer_height.max(h),
            },
        })
    }

    /// Whether `geom` would still keep the load within the container's mass.
    fn mass_allows(&self, geom: &Geometry3D) -> bool {
        match (self.boundary.max_mass(), geom.mass()) {
            (Some(max_mass), Some(mass)) => self.mass + mass <= max_mass,
            _ => true,
        }
    }

    fn commit(&mut self, geom: &Geometry3D, instance: usize, orientation: usize, spot: LayerSpot) {
        // The orientation goes out as `rotation_index`, so the response
        // reports the box's actual footprint.
        self.placements.push(
            Placement::new_3d(
                geom.id().clone(),
                instance,
                spot.x,
                spot.y,
                spot.z,
                0.0,
                0.0,
                0.0,
            )
            .with_rotation_index(orientation),
        );
        self.cursor = spot.next;
        self.volume += geom.measure();
        self.mass += geom.mass().unwrap_or(0.0);
    }

    /// Places `instance` of `geom` turned to `orientation`, if it fits there.
    pub fn place(&mut self, geom: &Geometry3D, instance: usize, orientation: usize) -> bool {
        if !self.mass_allows(geom) {
            return false;
        }
        let dims = geom.dimensions_for_orientation(orientation);
        match self.spot([dims.x, dims.y, dims.z]) {
            Some(spot) => {
                self.commit(geom, instance, orientation, spot);
                true
            }
            None => false,
        }
    }

    /// Places `instance` of `geom` in whichever allowed orientation keeps it
    /// lowest, then nearest the front, then nearest the left, then flattest
    /// (the first such orientation on a tie).
    pub fn place_best(&mut self, geom: &Geometry3D, instance: usize) -> bool {
        if !self.mass_allows(geom) {
            return false;
        }
        let score = |spot: &LayerSpot, height: f64| {
            spot.z * 1_000_000.0 + spot.y * 1000.0 + spot.x + height * 0.1
        };
        let mut best: Option<(usize, LayerSpot, f64)> = None;
        for orientation in 0..geom.allowed_orientations().len() {
            let dims = geom.dimensions_for_orientation(orientation);
            if let Some(spot) = self.spot([dims.x, dims.y, dims.z]) {
                let s = score(&spot, dims.z);
                if best.is_none_or(|(_, _, b)| s < b) {
                    best = Some((orientation, spot, s));
                }
            }
        }
        match best {
            Some((orientation, spot, _)) => {
                self.commit(geom, instance, orientation, spot);
                true
            }
            None => false,
        }
    }

    /// The boxes placed so far.
    pub fn placed(&self) -> usize {
        self.placements.len()
    }

    /// Fraction of the container volume the placed boxes fill.
    pub fn utilization(&self) -> f64 {
        self.volume / self.boundary.measure()
    }

    /// The placements, the utilization and the count placed.
    pub fn finish(self) -> LayerPlacementResult {
        LayerPlacementResult {
            utilization: self.utilization(),
            placed_count: self.placements.len(),
            placements: self.placements,
        }
    }
}

/// Places items in the given order and orientations, layer by layer (see
/// [`LayerFill`]). Shared by the GA, BRKGA and SA decoders.
///
/// An item that does not fit, by size or by the container's mass, is
/// skipped; a cancelled run stops placing.
pub fn layer_place_items(
    items: &[PlacementItem],
    instances: &[InstanceInfo],
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    cancelled: &Arc<AtomicBool>,
) -> LayerPlacementResult {
    let mut fill = LayerFill::new(boundary, config);
    for item in items {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let Some(info) = instances.get(item.instance_idx) else {
            continue;
        };
        let geom = &geometries[info.geometry_idx];
        let orientation = item.orientation_idx % info.orientation_count.max(1);
        fill.place(geom, info.instance_num, orientation);
    }
    fill.finish()
}

/// The layer fill: every instance in request order (each geometry's
/// instances in turn), each in its best orientation
/// ([`LayerFill::place_best`]).
///
/// `stop` is asked before each instance; when it says so the fill ends with
/// what it has placed, and the second value is `true`. `placed` hears of
/// every box placed.
pub fn layer_fill_in_order(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    mut stop: impl FnMut() -> bool,
    mut placed: impl FnMut(&LayerFill<'_>),
) -> (LayerPlacementResult, bool) {
    let mut fill = LayerFill::new(boundary, config);
    for geom in geometries {
        for instance in 0..geom.quantity() {
            if stop() {
                return (fill.finish(), true);
            }
            if fill.place_best(geom, instance) {
                placed(&fill);
            }
        }
    }
    (fill.finish(), false)
}

/// The orientation of each instance (in [`build_instances`] order) in
/// `placements`; 0 for an instance they do not contain. Placing the instances
/// in that order and those orientations with [`layer_place_items`] reproduces
/// a layout [`layer_fill_in_order`] produced -- how a search starts from it.
pub fn seed_orientations(
    instances: &[InstanceInfo],
    geometries: &[Geometry3D],
    placements: &[Placement<f64>],
) -> Vec<usize> {
    let placed: std::collections::HashMap<(&str, usize), usize> = placements
        .iter()
        .map(|p| {
            (
                (p.geometry_id.as_str(), p.instance),
                p.rotation_index.unwrap_or(0),
            )
        })
        .collect();
    instances
        .iter()
        .map(|info| {
            let id = geometries[info.geometry_idx].id().as_str();
            placed.get(&(id, info.instance_num)).copied().unwrap_or(0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_instances_single() {
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(3)];
        let instances = build_instances(&geometries);
        assert_eq!(instances.len(), 3);
        assert_eq!(instances[0].geometry_idx, 0);
        assert_eq!(instances[0].instance_num, 0);
        assert_eq!(instances[2].instance_num, 2);
    }

    #[test]
    fn test_build_instances_multiple() {
        let geometries = vec![
            Geometry3D::new("A", 10.0, 10.0, 10.0).with_quantity(2),
            Geometry3D::new("B", 20.0, 20.0, 20.0).with_quantity(3),
        ];
        let instances = build_instances(&geometries);
        assert_eq!(instances.len(), 5);
        assert_eq!(instances[0].geometry_idx, 0);
        assert_eq!(instances[1].geometry_idx, 0);
        assert_eq!(instances[2].geometry_idx, 1);
        assert_eq!(instances[4].geometry_idx, 1);
    }

    #[test]
    fn test_packing_fitness_all_placed() {
        let fitness = packing_fitness(10, 10, 0.8);
        // 1.0 * 100 + 0.8 * 10 = 108.0
        assert!((fitness - 108.0).abs() < 1e-10);
    }

    #[test]
    fn test_packing_fitness_partial() {
        let fitness = packing_fitness(5, 10, 0.5);
        // 0.5 * 100 + 0.5 * 10 = 55.0
        assert!((fitness - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_packing_fitness_none_placed() {
        let fitness = packing_fitness(0, 10, 0.0);
        assert!((fitness - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_packing_fitness_zero_total() {
        let fitness = packing_fitness(0, 0, 0.0);
        assert!((fitness - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_unplaced_list_all_placed() {
        let geometries = vec![
            Geometry3D::new("A", 10.0, 10.0, 10.0),
            Geometry3D::new("B", 20.0, 20.0, 20.0),
        ];
        let placements = vec![
            Placement::new_3d("A".to_string(), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Placement::new_3d("B".to_string(), 0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let unplaced = build_unplaced_list(&placements, &geometries);
        assert!(unplaced.is_empty());
    }

    #[test]
    fn test_build_unplaced_list_partial() {
        let geometries = vec![
            Geometry3D::new("A", 10.0, 10.0, 10.0),
            Geometry3D::new("B", 20.0, 20.0, 20.0),
        ];
        let placements = vec![Placement::new_3d(
            "A".to_string(),
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )];
        let unplaced = build_unplaced_list(&placements, &geometries);
        assert_eq!(unplaced, vec!["B"]);
    }

    #[test]
    fn test_layer_place_items_respects_bounds() {
        // Invariant: every placed box lies fully inside the boundary for its actual
        // resolved orientation, across mixed sizes and Fixed/Any constraints.
        use crate::geometry::OrientationConstraint::{Any, Fixed};
        let dims_pool = [
            (30.0, 20.0, 25.0),
            (40.0, 10.0, 40.0),
            (60.0, 70.0, 15.0),
            (15.0, 80.0, 35.0),
            (50.0, 30.0, 45.0),
            (90.0, 90.0, 10.0),
        ];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut seed: u64 = 0xABCD_1234;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for constraint in [Fixed, Any] {
            for _ in 0..200 {
                // Each geometry quantity 1 with a unique id → deterministic orientation map.
                let n = 3 + next() % 8;
                let orients: Vec<usize> = (0..n).map(|_| next()).collect();
                let geometries: Vec<_> = (0..n)
                    .map(|g| {
                        let (w, d, h) = dims_pool[next() % dims_pool.len()];
                        Geometry3D::new(format!("g{g}"), w, d, h).with_orientation(constraint)
                    })
                    .collect();
                let instances = build_instances(&geometries);
                let items: Vec<PlacementItem> = (0..instances.len())
                    .map(|i| PlacementItem {
                        instance_idx: i,
                        orientation_idx: orients[i],
                    })
                    .collect();
                let result = layer_place_items(
                    &items,
                    &instances,
                    &geometries,
                    &boundary,
                    &config,
                    &cancelled,
                );
                for p in &result.placements {
                    let gi: usize = p.geometry_id[1..].parse().unwrap();
                    let oc = geometries[gi].allowed_orientations().len().max(1);
                    let dm = geometries[gi].dimensions_for_orientation(orients[gi] % oc);
                    let (px, py, pz) = (p.x(), p.y(), p.z().unwrap_or(0.0));
                    let e = 1e-6;
                    assert!(
                        px + dm.x <= 100.0 + e && py + dm.y <= 100.0 + e && pz + dm.z <= 100.0 + e,
                        "{} out of bounds: pos({px:.1},{py:.1},{pz:.1}) dims({:.1},{:.1},{:.1})",
                        p.geometry_id,
                        dm.x,
                        dm.y,
                        dm.z,
                    );
                }
            }
        }
    }

    #[test]
    fn test_layer_place_items_basic() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let instances = build_instances(&geometries);
        let cancelled = Arc::new(AtomicBool::new(false));

        let items: Vec<PlacementItem> = (0..instances.len())
            .map(|i| PlacementItem {
                instance_idx: i,
                orientation_idx: 0,
            })
            .collect();

        let result = layer_place_items(
            &items,
            &instances,
            &geometries,
            &boundary,
            &config,
            &cancelled,
        );

        assert_eq!(result.placed_count, 2);
        assert_eq!(result.placements.len(), 2);
        assert!(result.utilization > 0.0);
    }

    #[test]
    fn test_layer_place_items_cancellation() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(5)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default();
        let instances = build_instances(&geometries);
        let cancelled = Arc::new(AtomicBool::new(true)); // Already cancelled

        let items: Vec<PlacementItem> = (0..instances.len())
            .map(|i| PlacementItem {
                instance_idx: i,
                orientation_idx: 0,
            })
            .collect();

        let result = layer_place_items(
            &items,
            &instances,
            &geometries,
            &boundary,
            &config,
            &cancelled,
        );

        assert_eq!(result.placed_count, 0);
    }

    #[test]
    fn test_layer_place_items_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0).with_max_mass(250.0);
        let config = Config::default();
        let instances = build_instances(&geometries);
        let cancelled = Arc::new(AtomicBool::new(false));

        let items: Vec<PlacementItem> = (0..instances.len())
            .map(|i| PlacementItem {
                instance_idx: i,
                orientation_idx: 0,
            })
            .collect();

        let result = layer_place_items(
            &items,
            &instances,
            &geometries,
            &boundary,
            &config,
            &cancelled,
        );

        // Max 2 boxes at 100kg each within 250kg limit
        assert!(result.placed_count <= 2);
    }

    /// A search starts from the layer fill by its genes: the instances in
    /// request order with the orientations the fill chose decode back to the
    /// fill's own layout, box for box.
    #[test]
    fn the_layer_fill_round_trips_through_the_decoder() {
        use crate::geometry::OrientationConstraint::Any;
        let geometries = vec![
            Geometry3D::new("A", 30.0, 20.0, 10.0)
                .with_quantity(3)
                .with_orientation(Any),
            Geometry3D::new("B", 45.0, 15.0, 15.0)
                .with_quantity(2)
                .with_orientation(Any),
            Geometry3D::new("C", 70.0, 10.0, 10.0).with_orientation(Any),
            Geometry3D::new("D", 25.0, 25.0, 25.0).with_quantity(4),
        ];
        let boundary = Boundary3D::new(60.0, 50.0, 40.0);
        let config = Config::default().with_spacing(1.0).with_margin(2.0);
        let (fill, stopped) =
            layer_fill_in_order(&geometries, &boundary, &config, || false, |_| {});
        assert!(!stopped);
        assert!(fill.placed_count > 0);

        let instances = build_instances(&geometries);
        let orientations = seed_orientations(&instances, &geometries, &fill.placements);
        let items: Vec<PlacementItem> = (0..instances.len())
            .map(|i| PlacementItem {
                instance_idx: i,
                orientation_idx: orientations[i],
            })
            .collect();
        let decoded = layer_place_items(
            &items,
            &instances,
            &geometries,
            &boundary,
            &config,
            &Arc::new(AtomicBool::new(false)),
        );
        let key = |p: &Placement<f64>| {
            (
                p.geometry_id.clone(),
                p.instance,
                p.position.clone(),
                p.rotation_index,
            )
        };
        assert_eq!(
            decoded.placements.iter().map(key).collect::<Vec<_>>(),
            fill.placements.iter().map(key).collect::<Vec<_>>()
        );
        assert_eq!(decoded.utilization.to_bits(), fill.utilization.to_bits());
    }

    /// A box that fits in no row or layer is skipped and leaves the fill
    /// where it was: it is neither placed out of the container nor pushes the
    /// boxes after it into a new layer.
    #[test]
    fn a_box_that_does_not_fit_is_skipped_without_moving_the_fill() {
        let geometries = vec![
            Geometry3D::new("small", 10.0, 10.0, 10.0),
            Geometry3D::new("too_wide", 120.0, 10.0, 10.0),
            Geometry3D::new("too_tall", 10.0, 10.0, 120.0),
            Geometry3D::new("small2", 10.0, 10.0, 10.0),
        ];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let instances = build_instances(&geometries);
        let items: Vec<PlacementItem> = (0..instances.len())
            .map(|i| PlacementItem {
                instance_idx: i,
                orientation_idx: 0,
            })
            .collect();
        let result = layer_place_items(
            &items,
            &instances,
            &geometries,
            &boundary,
            &Config::default(),
            &Arc::new(AtomicBool::new(false)),
        );
        let ids: Vec<&str> = result
            .placements
            .iter()
            .map(|p| p.geometry_id.as_str())
            .collect();
        assert_eq!(ids, ["small", "small2"]);
        assert_eq!(result.placements[1].position, vec![10.0, 0.0, 0.0]);
    }

    /// Two of three copies placed leaves the geometry unplaced.
    #[test]
    fn test_build_unplaced_list_counts_instances() {
        let geometries = vec![Geometry3D::new("A", 10.0, 10.0, 10.0).with_quantity(3)];
        let placements = vec![
            Placement::new_3d("A".to_string(), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Placement::new_3d("A".to_string(), 2, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        assert_eq!(build_unplaced_list(&placements, &geometries), vec!["A"]);
    }
}
