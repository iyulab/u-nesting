//! Every strategy reports its progress while it runs, and a caller that
//! cancels from its progress callback -- the only place a C or .NET caller
//! can -- stops the run there. Only the fills and the GA used to report, so a
//! cancel never reached the other searches.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use u_nesting_d2::{Boundary2D, Config, Geometry2D, Nester2D, Solver, StopReason, Strategy};

const STRATEGIES: [Strategy; 7] = [
    Strategy::BottomLeftFill,
    Strategy::NfpGuided,
    Strategy::GeneticAlgorithm,
    Strategy::Brkga,
    Strategy::SimulatedAnnealing,
    Strategy::Gdrr,
    Strategy::Alns,
];

/// Rectangles of eight sizes -- quick to place, so every search gets to its
/// own steps whatever the machine's load.
fn pieces(count_per_kind: usize) -> Vec<Geometry2D> {
    (0..8)
        .map(|i| {
            Geometry2D::rectangle(format!("R{i}"), 10.0 + 3.0 * i as f64, 6.0 + 2.0 * i as f64)
                .with_quantity(count_per_kind)
                .with_rotations_deg(vec![0.0, 90.0])
        })
        .collect()
}

/// The phase the strategy itself reports under -- a search's greedy start
/// reports as the NFP fill it is.
fn own_phase(strategy: Strategy) -> &'static str {
    match strategy {
        Strategy::BottomLeftFill => "BLF Placement",
        Strategy::NfpGuided => "NFP Placement",
        Strategy::GeneticAlgorithm => "Genetic Algorithm",
        Strategy::Brkga => "BRKGA",
        Strategy::SimulatedAnnealing => "Simulated Annealing",
        Strategy::Gdrr => "GDRR",
        Strategy::Alns => "ALNS",
        other => unreachable!("{other:?}"),
    }
}

fn config(strategy: Strategy, time_limit_ms: u64) -> Config {
    let config = Config::default()
        .with_strategy(strategy)
        .with_time_limit(time_limit_ms);
    if strategy.reads_setting(u_nesting_core::solver::Dimension::TwoD, "seed") {
        config.with_seed(1)
    } else {
        config
    }
}

/// The C ABI's shape of a cancel: the progress callback sets the solver's
/// flag, and the run returns what it has, saying it was cancelled. The cancel
/// is set on the strategy's own second report -- past any greedy start -- so a
/// strategy that does not report while it runs ends some other way and fails
/// here. A run reports once per placed piece, evaluated batch or search step,
/// so a cancel waits at most for one layout to be built.
#[test]
fn a_cancel_from_the_progress_callback_stops_every_strategy() {
    let pieces = pieces(3);
    let boundary = Boundary2D::rectangle(500.0, 5000.0);
    for strategy in STRATEGIES {
        let nester = Nester2D::new(config(strategy, 600_000));
        let cancel = nester.cancel_handle();
        let reports = Arc::new(AtomicUsize::new(0));
        let cancelled_at = Arc::new(Mutex::new(None));
        let (seen, at) = (reports.clone(), cancelled_at.clone());
        let result = nester
            .solve_with_progress(
                &pieces,
                &boundary,
                // Cancelled on the strategy's own second report: in the
                // search itself, past any greedy start.
                Box::new(move |info| {
                    if info.phase == own_phase(strategy)
                        && seen.fetch_add(1, Ordering::Relaxed) == 1
                    {
                        *at.lock().expect("not poisoned") = Some(Instant::now());
                        cancel.store(true, Ordering::Relaxed);
                    }
                }),
            )
            .expect("solved");
        assert_eq!(result.stop_reason, StopReason::Cancelled, "{strategy:?}");
        // After the cancel the run finishes the step it is in and decodes
        // the best layout it has -- one layout's work, not the rest of its
        // budget. The limit is far off (a search gets a quarter of it), so
        // the cancel is what ends the run however slow the machine.
        let after = cancelled_at
            .lock()
            .expect("not poisoned")
            .expect("cancelled")
            .elapsed();
        assert!(
            after < Duration::from_secs(10),
            "{strategy:?}: {after:?} from the cancel to the answer"
        );
    }
}

#[test]
fn a_multi_sheet_solve_reports_and_cancels_the_same_way() {
    let pieces = pieces(6);
    let boundary = Boundary2D::rectangle(150.0, 150.0);
    let nester = Nester2D::new(config(Strategy::Gdrr, 600_000));
    let cancel = nester.cancel_handle();
    let reports = Arc::new(AtomicUsize::new(0));
    let seen = reports.clone();
    let result = nester
        .solve_multi_strip_with_progress(
            &pieces,
            &boundary,
            Box::new(move |info| {
                if info.phase == "GDRR" && seen.fetch_add(1, Ordering::Relaxed) == 20 {
                    cancel.store(true, Ordering::Relaxed);
                }
            }),
        )
        .expect("solved");
    assert!(reports.load(Ordering::Relaxed) > 20);
    assert_eq!(result.stop_reason, StopReason::Cancelled);
}
