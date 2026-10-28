//! A packing does not depend on the unit the sizes are given in: the same
//! boxes and container in millimetres or metres, seeded alike, are packed the
//! same way. The searches score layouts by the share of boxes placed and the
//! volume they fill -- ratios -- and their temperatures and thresholds are on
//! that score, not on lengths.

use u_nesting_d3::geometry::OrientationConstraint;
use u_nesting_d3::{Boundary3D, Config, Geometry3D, Packer3D, SolveResult, Solver, Strategy};

fn request(scale: f64) -> (Vec<Geometry3D>, Boundary3D) {
    let any = OrientationConstraint::Any;
    let s = |v: f64| v * scale;
    (
        vec![
            Geometry3D::new("A", s(30.0), s(20.0), s(10.0))
                .with_quantity(3)
                .with_orientation(any),
            Geometry3D::new("B", s(25.0), s(15.0), s(15.0))
                .with_quantity(3)
                .with_orientation(any),
            Geometry3D::new("C", s(40.0), s(10.0), s(10.0))
                .with_quantity(2)
                .with_orientation(any),
        ],
        Boundary3D::new(s(60.0), s(50.0), s(40.0)),
    )
}

fn solve(strategy: Strategy, scale: f64) -> SolveResult<f64> {
    let (pieces, boundary) = request(scale);
    let config = Config::default()
        .with_strategy(strategy)
        .with_time_limit(0)
        .with_population_size(8)
        .with_max_generations(6)
        .with_seed(7);
    Packer3D::new(config)
        .solve(&pieces, &boundary)
        .expect("the request is valid")
}

#[test]
fn every_strategy_packs_the_same_in_any_unit() {
    for strategy in [
        Strategy::BottomLeftFill,
        Strategy::ExtremePoint,
        Strategy::GeneticAlgorithm,
        Strategy::Brkga,
        Strategy::SimulatedAnnealing,
    ] {
        let base = solve(strategy, 1.0);
        for scale in [1000.0, 0.01] {
            let scaled = solve(strategy, scale);
            assert_eq!(
                scaled.placements.len(),
                base.placements.len(),
                "{strategy:?} at {scale}: boxes placed"
            );
            for (a, b) in base.placements.iter().zip(&scaled.placements) {
                assert_eq!(
                    (&a.geometry_id, a.instance, a.rotation_index),
                    (&b.geometry_id, b.instance, b.rotation_index),
                    "{strategy:?} at {scale}: the same box in the same turn"
                );
                for (x, y) in a.position.iter().zip(&b.position) {
                    assert!(
                        (x * scale - y).abs() <= 1e-9 * scale * 100.0,
                        "{strategy:?} at {scale}: position {x} becomes {y}"
                    );
                }
            }
            assert!(
                (base.utilization - scaled.utilization).abs() < 1e-9,
                "{strategy:?} at {scale}: utilisation {} vs {}",
                base.utilization,
                scaled.utilization
            );
        }
    }
}
