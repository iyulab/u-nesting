//! What `Config::seed` promises for 2D nesting: a seeded run that ends on its
//! generation or iteration budget is reproduced exactly by the same request,
//! and the seed is what the search draws from.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use u_nesting_core::alns::AlnsConfig;
use u_nesting_core::gdrr::GdrrConfig;
use u_nesting_d2::alns_nesting::run_alns_nesting;
use u_nesting_d2::gdrr_nesting::run_gdrr_nesting;
use u_nesting_d2::{Boundary2D, Config, Geometry2D, Nester2D, SolveResult, Solver, Strategy};

const RUNS: usize = 4;

/// A placement reduced to what identifies it, compared bit for bit.
type Key = (String, usize, Vec<u64>, Vec<u64>, bool);

/// What a search returned: its layout and the fitness it reported.
type Found = (Vec<Key>, Option<u64>);

fn layout(result: &SolveResult<f64>) -> Vec<Key> {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect();
    result
        .placements
        .iter()
        .map(|p| {
            (
                p.geometry_id.clone(),
                p.instance,
                bits(&p.position),
                bits(&p.rotation),
                p.mirrored,
            )
        })
        .collect()
}

/// Pieces that may turn (and one that may flip), so orders and rotations
/// give many layouts.
fn mixed() -> (Vec<Geometry2D>, Boundary2D) {
    (
        vec![
            Geometry2D::rectangle("A", 30.0, 20.0)
                .with_rotations_deg(vec![0.0, 90.0])
                .with_quantity(2),
            Geometry2D::rectangle("B", 25.0, 15.0)
                .with_rotations_deg(vec![0.0, 90.0])
                .with_quantity(2),
            Geometry2D::new("T")
                .with_polygon(vec![(0.0, 0.0), (24.0, 0.0), (0.0, 18.0)])
                .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0])
                .with_flip(true),
        ],
        Boundary2D::rectangle(60.0, 50.0),
    )
}

/// The whole solve, for the searches whose budget a request sets: a
/// generation budget the run ends on, inside a time limit it never reaches.
#[test]
fn the_same_seed_gives_the_same_layout() {
    let (pieces, boundary) = mixed();
    for strategy in [
        Strategy::GeneticAlgorithm,
        Strategy::Brkga,
        Strategy::SimulatedAnnealing,
    ] {
        let solve = || {
            Nester2D::new(
                Config::default()
                    .with_strategy(strategy)
                    .with_time_limit(600_000)
                    .with_population_size(6)
                    .with_max_generations(3)
                    .with_seed(11),
            )
            .solve(&pieces, &boundary)
            .expect("solve")
        };
        let first = solve();
        assert!(!first.placements.is_empty(), "{strategy:?}");
        for run in 1..RUNS {
            assert_eq!(
                layout(&solve()),
                layout(&first),
                "{strategy:?}: run {run} differs from run 0 with the same seed"
            );
        }
    }
}

/// GDRR and ALNS run a fixed iteration count no request sets, so they are
/// run here on a short count, unfloored -- what they found, not the greedy
/// layout a solve might return instead. (That the solve hands them
/// `config.seed` is a unit test of the nester.)
#[test]
fn ruin_and_recreate_searches_draw_from_the_seed() {
    // Pieces of three shapes in many sizes, each turning four ways, on a sheet
    // they roughly fill: the layout the greedy start gives can be improved
    // several ways, so a ruin has choices that matter. (Rectangles do not do:
    // the greedy start is the best layout the search finds, whatever the seed.)
    let pieces: Vec<Geometry2D> = (0..10)
        .map(|i| {
            let k = i as f64;
            let (w, h) = (10.0 + (k * 7.0) % 17.0, 12.0 + (k * 5.0) % 13.0);
            let outline = match i % 3 {
                0 => vec![(0.0, 0.0), (w, 0.0), (0.0, h)],
                1 => vec![
                    (0.0, 0.0),
                    (w, 0.0),
                    (w, h * 0.4),
                    (w * 0.4, h * 0.4),
                    (w * 0.4, h),
                    (0.0, h),
                ],
                _ => vec![(0.0, 0.0), (w, 0.0), (w * 0.8, h), (w * 0.2, h)],
            };
            Geometry2D::new(format!("P{i}"))
                .with_polygon(outline)
                .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0])
        })
        .collect();
    let boundary = Boundary2D::rectangle(70.0, 50.0);
    let config = Config::default();
    let gdrr = |seed: u64| {
        let run = GdrrConfig::new()
            .with_max_iterations(25)
            .with_time_limit_ms(0)
            .with_seed(seed);
        let r = run_gdrr_nesting(
            &pieces,
            &boundary,
            &config,
            &run,
            Arc::new(AtomicBool::new(false)),
            None,
        )
        .expect("valid settings");
        (layout(&r), r.best_fitness.map(f64::to_bits))
    };
    let alns = |seed: u64| {
        let run = AlnsConfig::new()
            .with_max_iterations(25)
            .with_time_limit_ms(0)
            .with_seed(seed);
        let r = run_alns_nesting(
            &pieces,
            &boundary,
            &config,
            &run,
            Arc::new(AtomicBool::new(false)),
            None,
        )
        .expect("valid settings");
        (layout(&r), r.best_fitness.map(f64::to_bits))
    };
    let searches: [(&str, &dyn Fn(u64) -> Found); 2] = [("GDRR", &gdrr), ("ALNS", &alns)];
    for (name, search) in searches {
        let first = search(3);
        for run in 1..RUNS {
            assert_eq!(
                search(3),
                first,
                "{name}: run {run} differs with the same seed"
            );
        }
        let distinct: std::collections::HashSet<_> = (0..6).map(search).collect();
        assert!(distinct.len() > 1, "{name}: six seeds gave one result");
    }
}
