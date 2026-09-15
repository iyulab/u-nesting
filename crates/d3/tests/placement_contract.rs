//! The placement contract every 3D strategy shares, measured on the output:
//! `spacing` is the minimum gap between any two placed boxes, and `margin` is
//! the minimum gap between any box and a container wall.

use proptest::prelude::*;
use u_nesting_core::Geometry;
use u_nesting_d3::geometry::OrientationConstraint;
use u_nesting_d3::{Boundary3D, Config, Geometry3D, Packer3D, SolveResult, Solver, Strategy};

const STRATEGIES: [Strategy; 5] = [
    Strategy::BottomLeftFill,
    Strategy::ExtremePoint,
    Strategy::GeneticAlgorithm,
    Strategy::Brkga,
    Strategy::SimulatedAnnealing,
];

const TOL: f64 = 1e-6;

fn config(strategy: Strategy) -> Config {
    Config::default()
        .with_strategy(strategy)
        .with_time_limit(1000)
        .with_seed(3)
}

/// Placed boxes as `(min, max)` corners. Every piece here has a fixed
/// orientation, so its extent is its own dimensions.
fn boxes(pieces: &[Geometry3D], result: &SolveResult<f64>) -> Vec<([f64; 3], [f64; 3])> {
    result
        .placements
        .iter()
        .map(|p| {
            let g = pieces
                .iter()
                .find(|g| g.id() == &p.geometry_id)
                .expect("placement refers to a requested piece");
            let d = g.dimensions();
            let min = [p.position[0], p.position[1], p.position[2]];
            (min, [min[0] + d.x, min[1] + d.y, min[2] + d.z])
        })
        .collect()
}

/// Euclidean gap between two axis-aligned boxes (0 when they overlap).
fn box_gap(a: &([f64; 3], [f64; 3]), b: &([f64; 3], [f64; 3])) -> f64 {
    (0..3)
        .map(|k| (b.0[k] - a.1[k]).max(a.0[k] - b.1[k]).max(0.0))
        .map(|g| g * g)
        .sum::<f64>()
        .sqrt()
}

fn overlaps(a: &([f64; 3], [f64; 3]), b: &([f64; 3], [f64; 3])) -> bool {
    (0..3).all(|k| a.0[k] < b.1[k] - TOL && b.0[k] < a.1[k] - TOL)
}

fn min_gap(placed: &[([f64; 3], [f64; 3])]) -> f64 {
    let mut gap = f64::INFINITY;
    for i in 0..placed.len() {
        for j in i + 1..placed.len() {
            gap = gap.min(box_gap(&placed[i], &placed[j]));
        }
    }
    gap
}

fn min_wall_clearance(placed: &[([f64; 3], [f64; 3])], size: [f64; 3]) -> f64 {
    placed
        .iter()
        .flat_map(|(min, max)| (0..3).flat_map(move |k| [min[k], size[k] - max[k]]))
        .fold(f64::INFINITY, f64::min)
}

fn cube(id: &str, side: f64, quantity: usize) -> Geometry3D {
    Geometry3D::new(id, side, side, side)
        .with_quantity(quantity)
        .with_orientation(OrientationConstraint::Fixed)
}

/// Eight 40-unit cubes fit a 100-unit container with a 5-unit gap
/// (40 + 5 + 40 = 85 ≤ 100 on every axis) — spacing must not cost boxes.
#[test]
fn spacing_does_not_cost_boxes_that_fit() {
    let pieces = [cube("B", 40.0, 8)];
    let boundary = Boundary3D::new(100.0, 100.0, 100.0);
    for strategy in STRATEGIES {
        let result = Packer3D::new(config(strategy).with_spacing(5.0))
            .solve(&pieces, &boundary)
            .unwrap();
        let placed = boxes(&pieces, &result);
        assert_eq!(placed.len(), 8, "{strategy:?}: placed {}", placed.len());
        let gap = min_gap(&placed);
        assert!(gap >= 5.0 - TOL, "{strategy:?}: gap {gap}");
    }
}

/// A single box sits `margin` from the three walls at the origin.
#[test]
fn a_single_box_sits_at_the_margin_corner() {
    let pieces = [cube("B", 10.0, 1)];
    let boundary = Boundary3D::new(100.0, 100.0, 100.0);
    for strategy in STRATEGIES {
        for margin in [0.0, 5.0, 20.0] {
            let result = Packer3D::new(config(strategy).with_margin(margin))
                .solve(&pieces, &boundary)
                .unwrap();
            assert_eq!(result.placements.len(), 1, "{strategy:?} margin={margin}");
            let p = &result.placements[0].position;
            assert!(
                (0..3).all(|k| (p[k] - margin).abs() < TOL),
                "{strategy:?} margin={margin}: placed at {p:?}"
            );
        }
    }
}

/// With a margin that leaves exactly room for a 2 × 2 × 2 stack, all eight fit.
#[test]
fn a_tight_margin_still_admits_what_fits() {
    let pieces = [cube("B", 10.0, 8)];
    let boundary = Boundary3D::new(40.0, 40.0, 40.0);
    for strategy in STRATEGIES {
        let result = Packer3D::new(config(strategy).with_margin(10.0))
            .solve(&pieces, &boundary)
            .unwrap();
        let placed = boxes(&pieces, &result);
        assert_eq!(
            placed.len(),
            8,
            "{strategy:?}: unplaced {:?}",
            result.unplaced
        );
        assert!(min_wall_clearance(&placed, [40.0; 3]) >= 10.0 - TOL);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn every_packing_keeps_spacing_and_margin(
        dims in prop::collection::vec((5.0f64..40.0, 5.0f64..40.0, 5.0f64..40.0), 1..4),
        quantity in 1usize..5,
        spacing in 0.0f64..8.0,
        margin in 0.0f64..8.0,
        strategy in prop::sample::select(STRATEGIES.to_vec()),
    ) {
        let pieces: Vec<Geometry3D> = dims
            .iter()
            .enumerate()
            .map(|(i, &(w, d, h))| {
                Geometry3D::new(format!("B{i}"), w, d, h)
                    .with_quantity(quantity)
                    .with_orientation(OrientationConstraint::Fixed)
            })
            .collect();
        let size = [120.0, 100.0, 90.0];
        let result = Packer3D::new(
            config(strategy).with_spacing(spacing).with_margin(margin).with_time_limit(300),
        )
        .solve(&pieces, &Boundary3D::new(size[0], size[1], size[2]))
        .unwrap();
        let placed = boxes(&pieces, &result);
        for i in 0..placed.len() {
            for j in i + 1..placed.len() {
                prop_assert!(!overlaps(&placed[i], &placed[j]), "{strategy:?}: boxes {i} and {j} overlap");
            }
        }
        let gap = min_gap(&placed);
        prop_assert!(gap >= spacing - TOL, "{strategy:?}: gap {gap} < spacing {spacing}");
        let wall = min_wall_clearance(&placed, size);
        prop_assert!(wall >= margin - TOL, "{strategy:?}: wall {wall} < margin {margin}");
    }
}
