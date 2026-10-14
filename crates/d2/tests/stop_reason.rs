//! A solve says why it stopped. A layout cut short by the time limit and one
//! the strategy finished look alike from their placements; `stop_reason` is
//! the run's own account, set where each strategy stops.

use std::time::{Duration, Instant};

use u_nesting_d2::check::{check_layout, LayoutRules};
use u_nesting_d2::{Boundary2D, Config, Geometry2D, Nester2D, Solver, StopReason, Strategy};

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

fn single_sheet() -> LayoutRules {
    LayoutRules {
        spacing: 0.0,
        margin: 0.0,
        multi_sheet: false,
    }
}

#[test]
fn a_fill_that_tries_every_piece_has_completed() {
    let pieces = [Geometry2D::rectangle("sq", 10.0, 10.0).with_quantity(6)];
    let boundary = Boundary2D::rectangle(100.0, 100.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::NfpGuided] {
        let result = Nester2D::new(Config::default().with_strategy(strategy))
            .solve(&pieces, &boundary)
            .expect("solved");
        assert_eq!(result.stop_reason, StopReason::Completed, "{strategy:?}");
        assert!(result.all_placed(), "{strategy:?}");
    }
}

/// A fill stopped by the time limit returns the pieces it placed, and the
/// layout it reports is a valid one: `unplaced` names what was left.
#[test]
fn a_fill_cut_short_by_the_time_limit_keeps_what_it_placed() {
    let pieces = l_pieces(400);
    let total: usize = 8 * 400;
    let boundary = Boundary2D::rectangle(500.0, 500_000.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::NfpGuided] {
        let result = Nester2D::new(Config::default().with_strategy(strategy).with_time_limit(1))
            .solve(&pieces, &boundary)
            .expect("solved");
        assert_eq!(result.stop_reason, StopReason::TimeLimit, "{strategy:?}");
        assert!(
            result.placements.len() < total,
            "{strategy:?}: placed {} of {total} in 1 ms",
            result.placements.len()
        );
        let violations = check_layout(
            &pieces,
            &boundary,
            &single_sheet(),
            &result.placements,
            &result.unplaced,
        );
        assert!(violations.is_empty(), "{strategy:?}: {violations:?}");
    }
}

/// With no limit and a small budget, each search ends on its budget.
#[test]
fn a_search_ends_on_its_budget() {
    let pieces = [Geometry2D::rectangle("sq", 10.0, 10.0).with_quantity(6)];
    let boundary = Boundary2D::rectangle(100.0, 100.0);
    for strategy in [Strategy::GeneticAlgorithm, Strategy::Brkga] {
        let result = Nester2D::new(
            Config::default()
                .with_strategy(strategy)
                .with_time_limit(0)
                .with_population_size(6)
                .with_max_generations(3)
                .with_seed(1),
        )
        .solve(&pieces, &boundary)
        .expect("solved");
        assert_eq!(
            result.stop_reason,
            StopReason::IterationLimit,
            "{strategy:?}"
        );
    }
    // SA, GDRR and ALNS carry their own iteration caps.
    for strategy in [Strategy::SimulatedAnnealing, Strategy::Gdrr, Strategy::Alns] {
        let result = Nester2D::new(
            Config::default()
                .with_strategy(strategy)
                .with_time_limit(0)
                .with_seed(1),
        )
        .solve(&pieces, &boundary)
        .expect("solved");
        assert_eq!(
            result.stop_reason,
            StopReason::IterationLimit,
            "{strategy:?}"
        );
    }
}

#[test]
fn a_search_cut_short_by_the_time_limit_says_so() {
    let pieces = l_pieces(10);
    let boundary = Boundary2D::rectangle(500.0, 5000.0);
    for strategy in [
        Strategy::GeneticAlgorithm,
        Strategy::Brkga,
        Strategy::SimulatedAnnealing,
        Strategy::Gdrr,
        Strategy::Alns,
    ] {
        let result = Nester2D::new(
            Config::default()
                .with_strategy(strategy)
                .with_time_limit(300)
                .with_seed(1),
        )
        .solve(&pieces, &boundary)
        .expect("solved");
        assert_eq!(result.stop_reason, StopReason::TimeLimit, "{strategy:?}");
    }
}

/// Cancelling a running search stops it. GDRR and ALNS used to run on to
/// their iteration cap or the time limit.
#[test]
fn a_cancelled_search_stops_and_says_so() {
    let pieces = l_pieces(10);
    let boundary = Boundary2D::rectangle(500.0, 5000.0);
    for strategy in [
        Strategy::GeneticAlgorithm,
        Strategy::SimulatedAnnealing,
        Strategy::Gdrr,
        Strategy::Alns,
    ] {
        let nester = Nester2D::new(
            Config::default()
                .with_strategy(strategy)
                .with_time_limit(120_000)
                .with_seed(1),
        );
        let started = Instant::now();
        let result = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                nester.cancel();
            });
            nester.solve(&pieces, &boundary).expect("solved")
        });
        let took = started.elapsed();
        assert_eq!(result.stop_reason, StopReason::Cancelled, "{strategy:?}");
        assert!(
            took < Duration::from_secs(20),
            "{strategy:?}: {took:?} after a cancel at 300 ms"
        );
    }
}

/// The time limit bounds a multi-sheet solve as a whole: what is left of it is
/// shared among the sheets still needed. Every sheet used to get the full
/// limit afresh -- ten sheets ran ten times over.
#[test]
fn a_multi_sheet_solve_is_bounded_by_one_time_limit() {
    let pieces = l_pieces(20);
    let boundary = Boundary2D::rectangle(150.0, 150.0);
    let limit = 400;
    let started = Instant::now();
    let result = Nester2D::new(
        Config::default()
            .with_strategy(Strategy::GeneticAlgorithm)
            .with_time_limit(limit)
            .with_seed(1),
    )
    .solve_multi_strip(&pieces, &boundary)
    .expect("solved");
    let took = started.elapsed();
    assert_eq!(result.stop_reason, StopReason::TimeLimit);
    assert!(
        took < Duration::from_millis(limit) + Duration::from_millis(1_500),
        "{took:?} for a {limit} ms limit over {} sheets",
        result.boundaries_used
    );
    assert!(result.computation_time_ms > 0, "the run's time is recorded");
    // The limit is shared among the sheets rather than spent on the first.
    assert!(
        result.boundaries_used > 1,
        "{} sheet(s), {} of {} placed",
        result.boundaries_used,
        result.placements.len(),
        8 * 20
    );
}

/// `max_iterations` is the budget of SA, GDRR and ALNS: a seeded run that ends
/// on it, not on the clock, is the same run every time.
#[test]
fn an_iteration_budget_ends_a_seeded_run_the_same_way_every_time() {
    let pieces = l_pieces(2);
    let boundary = Boundary2D::rectangle(500.0, 5000.0);
    for strategy in [Strategy::SimulatedAnnealing, Strategy::Gdrr, Strategy::Alns] {
        let run = || {
            Nester2D::new(
                Config::default()
                    .with_strategy(strategy)
                    .with_time_limit(0)
                    .with_max_iterations(20)
                    .with_seed(3),
            )
            .solve(&pieces, &boundary)
            .expect("solved")
        };
        let (a, b) = (run(), run());
        assert_eq!(a.stop_reason, StopReason::IterationLimit, "{strategy:?}");
        assert!(
            a.iterations.is_some_and(|n| n <= 20),
            "{strategy:?}: {:?}",
            a.iterations
        );
        let layout = |r: &u_nesting_d2::SolveResult<f64>| {
            r.placements
                .iter()
                .map(|p| {
                    (
                        p.geometry_id.clone(),
                        p.instance,
                        p.position.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(layout(&a), layout(&b), "{strategy:?}");
    }
}
