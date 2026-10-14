//! A packing says why it stopped (`stop_reason`), set where each strategy
//! stops: a fill that tried every box, a search on its budget, a run cut short
//! by the time limit.

use u_nesting_d3::check::{check_packing, PackingRules};
use u_nesting_d3::{Boundary3D, Config, Geometry3D, Packer3D, Solver, StopReason, Strategy};

#[test]
fn a_fill_that_tries_every_box_has_completed() {
    let boxes = [Geometry3D::new("box", 2.0, 2.0, 2.0).with_quantity(8)];
    let container = Boundary3D::new(10.0, 10.0, 10.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::ExtremePoint] {
        let result = Packer3D::new(Config::default().with_strategy(strategy))
            .solve(&boxes, &container)
            .expect("solved");
        assert_eq!(result.stop_reason, StopReason::Completed, "{strategy:?}");
        assert!(result.all_placed(), "{strategy:?}");
    }
}

/// A fill stopped by the time limit returns the boxes it packed, and
/// `unplaced` names what was left.
#[test]
fn a_fill_cut_short_by_the_time_limit_keeps_what_it_packed() {
    let boxes: Vec<Geometry3D> = (0..6)
        .map(|i| {
            Geometry3D::new(
                format!("b{i}"),
                1.0 + 0.3 * i as f64,
                1.5,
                1.0 + 0.2 * i as f64,
            )
            .with_quantity(400)
        })
        .collect();
    let total = 6 * 400;
    let container = Boundary3D::new(60.0, 60.0, 60.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::ExtremePoint] {
        let result = Packer3D::new(Config::default().with_strategy(strategy).with_time_limit(1))
            .solve(&boxes, &container)
            .expect("solved");
        assert_eq!(result.stop_reason, StopReason::TimeLimit, "{strategy:?}");
        assert!(
            result.placements.len() < total,
            "{strategy:?}: packed {} of {total} in 1 ms",
            result.placements.len()
        );
        let rules = PackingRules {
            spacing: 0.0,
            margin: 0.0,
        };
        let violations = check_packing(
            &boxes,
            &container,
            &rules,
            &result.placements,
            &result.unplaced,
        );
        assert!(violations.is_empty(), "{strategy:?}: {violations:?}");
    }
}

/// With no limit and no target, each search ends on its own budget: GA and
/// BRKGA on generations (or a stall), SA when it has cooled.
#[test]
fn a_search_ends_on_its_budget() {
    let boxes = [Geometry3D::new("box", 2.0, 2.0, 2.0).with_quantity(4)];
    let container = Boundary3D::new(10.0, 10.0, 10.0);
    for strategy in [
        Strategy::GeneticAlgorithm,
        Strategy::Brkga,
        Strategy::SimulatedAnnealing,
    ] {
        let result = Packer3D::new(
            Config::default()
                .with_strategy(strategy)
                .with_time_limit(0)
                .with_seed(1),
        )
        .solve(&boxes, &container)
        .expect("solved");
        assert!(
            matches!(
                result.stop_reason,
                StopReason::IterationLimit | StopReason::NoImprovement | StopReason::Completed
            ),
            "{strategy:?}: {:?}",
            result.stop_reason
        );
    }
}
