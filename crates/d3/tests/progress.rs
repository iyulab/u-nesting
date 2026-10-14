//! Every 3D strategy reports its progress while it runs, and a caller that
//! cancels from its progress callback stops the run there. Only the layer
//! fill used to report.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use u_nesting_d3::{Boundary3D, Config, Geometry3D, Packer3D, Solver, StopReason, Strategy};

const STRATEGIES: [Strategy; 5] = [
    Strategy::BottomLeftFill,
    Strategy::ExtremePoint,
    Strategy::GeneticAlgorithm,
    Strategy::Brkga,
    Strategy::SimulatedAnnealing,
];

/// The phase the strategy itself reports under -- a search's layer fill
/// reports as the fill it is.
fn own_phase(strategy: Strategy) -> &'static str {
    match strategy {
        Strategy::BottomLeftFill => "Layer Packing",
        Strategy::ExtremePoint => "Extreme Point",
        Strategy::GeneticAlgorithm => "Genetic Algorithm",
        Strategy::Brkga => "BRKGA",
        Strategy::SimulatedAnnealing => "Simulated Annealing",
        other => unreachable!("{other:?}"),
    }
}

fn boxes(count_per_kind: usize) -> Vec<Geometry3D> {
    (0..5)
        .map(|i| {
            Geometry3D::new(format!("b{i}"), 2.0 + i as f64, 3.0, 1.5 + 0.5 * i as f64)
                .with_quantity(count_per_kind)
        })
        .collect()
}

fn config(strategy: Strategy, time_limit_ms: u64) -> Config {
    let config = Config::default()
        .with_strategy(strategy)
        .with_time_limit(time_limit_ms);
    if strategy.reads_setting(u_nesting_core::solver::Dimension::ThreeD, "seed") {
        config.with_seed(1)
    } else {
        config
    }
}

/// The cancel is set on the strategy's own second report -- past its layer
/// fill -- so a strategy that does not report while it runs ends some other
/// way and fails here.
#[test]
fn a_cancel_from_the_progress_callback_stops_every_strategy() {
    let boxes = boxes(40);
    let container = Boundary3D::new(30.0, 30.0, 30.0);
    for strategy in STRATEGIES {
        let packer = Packer3D::new(config(strategy, 600_000));
        let cancel = packer.cancel_handle();
        let reports = Arc::new(AtomicUsize::new(0));
        let seen = reports.clone();
        let result = packer
            .solve_with_progress(
                &boxes,
                &container,
                Box::new(move |info| {
                    if info.phase == own_phase(strategy)
                        && seen.fetch_add(1, Ordering::Relaxed) == 1
                    {
                        cancel.store(true, Ordering::Relaxed);
                    }
                }),
            )
            .expect("solved");
        assert_eq!(result.stop_reason, StopReason::Cancelled, "{strategy:?}");
    }
}
