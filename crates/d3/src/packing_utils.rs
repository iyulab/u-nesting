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

/// Builds a list of geometry IDs that were not placed.
///
/// Compares placed geometry IDs against the full input set to identify
/// which geometries have no placements.
pub fn build_unplaced_list(
    placements: &[Placement<f64>],
    geometries: &[Geometry3D],
) -> Vec<String> {
    let mut placed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in placements {
        placed_ids.insert(p.geometry_id.clone());
    }
    let mut unplaced = Vec::new();
    for geom in geometries {
        if !placed_ids.contains(geom.id()) {
            unplaced.push(geom.id().clone());
        }
    }
    unplaced
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

/// Core layer-based placement algorithm shared by all 3D packing solvers.
///
/// Places items in a layer-by-layer, row-by-row fashion within the boundary.
/// Items are placed left-to-right in rows, rows stack front-to-back in layers,
/// and layers stack bottom-to-top.
///
/// # Algorithm
///
/// For each item in the given order:
/// 1. Get oriented dimensions
/// 2. Check mass constraint
/// 3. Try to fit in current row (x direction)
/// 4. If not, move to next row (y direction)
/// 5. If row doesn't fit, move to next layer (z direction)
/// 6. If layer doesn't fit, skip item
///
/// # Arguments
///
/// * `items` - Ordered list of items with resolved orientations
/// * `instances` - Instance mapping (geometry index, instance number)
/// * `geometries` - Source geometry definitions
/// * `boundary` - Container dimensions and constraints
/// * `config` - Margin and spacing configuration
/// * `cancelled` - Cancellation flag for early termination
pub fn layer_place_items(
    items: &[PlacementItem],
    instances: &[InstanceInfo],
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    config: &Config,
    cancelled: &Arc<AtomicBool>,
) -> LayerPlacementResult {
    let mut placements = Vec::new();

    let margin = config.margin;
    let spacing = config.spacing;

    let bound_max_x = boundary.width() - margin;
    let bound_max_y = boundary.depth() - margin;
    let bound_max_z = boundary.height() - margin;

    let mut current_x = margin;
    let mut current_y = margin;
    let mut current_z = margin;
    let mut row_depth = 0.0_f64;
    let mut layer_height = 0.0_f64;

    let mut total_placed_volume = 0.0;
    let mut total_placed_mass = 0.0;
    let mut placed_count = 0;

    for item in items {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }

        if item.instance_idx >= instances.len() {
            continue;
        }

        let info = &instances[item.instance_idx];
        let geom = &geometries[info.geometry_idx];

        let orientation_idx = item.orientation_idx % info.orientation_count.max(1);

        // Get dimensions for this orientation
        let dims = geom.dimensions_for_orientation(orientation_idx);
        let g_width = dims.x;
        let g_depth = dims.y;
        let g_height = dims.z;

        // Check mass constraint
        if let (Some(max_mass), Some(item_mass)) = (boundary.max_mass(), geom.mass()) {
            if total_placed_mass + item_mass > max_mass {
                continue;
            }
        }

        // Try to fit in current row
        if current_x + g_width > bound_max_x {
            current_x = margin;
            current_y += row_depth + spacing;
            row_depth = 0.0;
        }

        // Check if fits in current layer (y direction)
        if current_y + g_depth > bound_max_y {
            current_x = margin;
            current_y = margin;
            current_z += layer_height + spacing;
            row_depth = 0.0;
            layer_height = 0.0;
        }

        // Check if fits in container height
        if current_z + g_height > bound_max_z {
            continue;
        }

        // Place the item. Preserve the resolved orientation via `rotation_index` so the
        // response reports the actual box footprint — without it, every rotated GA/BRKGA/SA
        // placement reported the identity orientation and read as out-of-bounds downstream.
        let placement = Placement::new_3d(
            geom.id().clone(),
            info.instance_num,
            current_x,
            current_y,
            current_z,
            0.0,
            0.0,
            0.0,
        )
        .with_rotation_index(orientation_idx);

        placements.push(placement);
        total_placed_volume += geom.measure();
        if let Some(mass) = geom.mass() {
            total_placed_mass += mass;
        }
        placed_count += 1;

        // Update position for next item
        current_x += g_width + spacing;
        row_depth = row_depth.max(g_depth);
        layer_height = layer_height.max(g_height);
    }

    let utilization = total_placed_volume / boundary.measure();
    LayerPlacementResult {
        placements,
        utilization,
        placed_count,
    }
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
}
