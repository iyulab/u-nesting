//! `time_limit_ms` is a limit on the whole solve, not a hint: every strategy
//! returns within it (plus the greedy baseline it is compared against), and a
//! solve leaves no threads behind.

use std::time::{Duration, Instant};

use u_nesting_d2::{Boundary2D, Config, Geometry2D, Nester2D, Solver, Strategy};

const STRATEGIES: [Strategy; 5] = [
    Strategy::GeneticAlgorithm,
    Strategy::Brkga,
    Strategy::SimulatedAnnealing,
    Strategy::Gdrr,
    Strategy::Alns,
];

fn l_pieces(count_per_kind: usize) -> Vec<Geometry2D> {
    (0..8)
        .map(|i| {
            let (w, h, t) = (30.0 + 6.0 * i as f64, 25.0 + 4.0 * i as f64, 10.0);
            Geometry2D::new(format!("L{i}"))
                .with_polygon(vec![(0.0, 0.0), (w, 0.0), (w, t), (t, t), (t, h), (0.0, h)])
                .with_quantity(count_per_kind)
                .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0])
        })
        .collect()
}

/// A search whose single placement pass costs more than the limit used to run
/// for a whole population of such passes — tens of seconds for a one-second
/// request. The limit now holds, and every piece is still placed.
#[test]
fn every_search_strategy_returns_within_its_time_limit() {
    let pieces = l_pieces(10);
    let boundary = Boundary2D::rectangle(500.0, 5000.0);
    let limit = Duration::from_millis(1000);

    // What the solve adds on top of the search: the greedy baseline it is
    // compared against, and the final boundary check.
    let started = Instant::now();
    Nester2D::new(Config::default().with_strategy(Strategy::BottomLeftFill))
        .solve(&pieces, &boundary)
        .unwrap();
    let baseline = started.elapsed();

    for strategy in STRATEGIES {
        let started = Instant::now();
        let result = Nester2D::new(
            Config::default()
                .with_strategy(strategy)
                .with_time_limit(limit.as_millis() as u64)
                .with_seed(1),
        )
        .solve(&pieces, &boundary)
        .unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed <= limit + baseline * 2 + Duration::from_millis(500),
            "{strategy:?}: {elapsed:?} for a {limit:?} limit"
        );
        assert_eq!(result.placements.len(), 80, "{strategy:?}");
    }
}

/// Each solve used to start a thread that waited for a cancellation that
/// never came.
#[cfg(target_os = "linux")]
#[test]
fn a_solve_leaves_no_threads_behind() {
    fn threads() -> usize {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix("Threads:"))
            .and_then(|n| n.trim().parse().ok())
            .unwrap()
    }
    let pieces = [Geometry2D::rectangle("r", 50.0, 30.0).with_quantity(6)];
    let boundary = Boundary2D::rectangle(300.0, 300.0);
    // Let the test harness and any worker pool settle first.
    Nester2D::new(
        Config::default()
            .with_strategy(Strategy::GeneticAlgorithm)
            .with_time_limit(100),
    )
    .solve(&pieces, &boundary)
    .unwrap();
    let before = threads();
    for strategy in [Strategy::GeneticAlgorithm, Strategy::SimulatedAnnealing] {
        for _ in 0..5 {
            Nester2D::new(
                Config::default()
                    .with_strategy(strategy)
                    .with_time_limit(100),
            )
            .solve(&pieces, &boundary)
            .unwrap();
        }
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        threads() <= before,
        "threads grew from {before} to {}",
        threads()
    );
}
