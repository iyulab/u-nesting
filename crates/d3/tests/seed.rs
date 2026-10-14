//! What `Config::seed` promises for 3D packing: a seeded run that ends on its
//! generation or iteration budget is reproduced exactly by the same request,
//! and no search returns a packing worse than the simple fill it starts from.

use u_nesting_d3::geometry::OrientationConstraint;
use u_nesting_d3::{Boundary3D, Config, Geometry3D, Packer3D, SolveResult, Solver, Strategy};

/// The strategies that draw random numbers.
const SEARCHES: [Strategy; 3] = [
    Strategy::GeneticAlgorithm,
    Strategy::Brkga,
    Strategy::SimulatedAnnealing,
];

const RUNS: usize = 12;

/// A placement reduced to what identifies it, compared bit for bit.
type Key = (String, usize, Vec<u64>, Option<usize>);

fn layout(result: &SolveResult<f64>) -> Vec<Key> {
    result
        .placements
        .iter()
        .map(|p| {
            (
                p.geometry_id.clone(),
                p.instance,
                p.position.iter().map(|v| v.to_bits()).collect(),
                p.rotation_index,
            )
        })
        .collect()
}

/// Mixed boxes that may turn, so orders and orientations give many layouts.
fn mixed() -> (Vec<Geometry3D>, Boundary3D) {
    let any = OrientationConstraint::Any;
    (
        vec![
            Geometry3D::new("A", 30.0, 20.0, 10.0)
                .with_quantity(3)
                .with_orientation(any),
            Geometry3D::new("B", 25.0, 15.0, 15.0)
                .with_quantity(3)
                .with_orientation(any),
            Geometry3D::new("C", 40.0, 10.0, 10.0)
                .with_quantity(2)
                .with_orientation(any),
        ],
        Boundary3D::new(60.0, 50.0, 40.0),
    )
}

/// A budget the run ends on, well inside a time limit it never reaches.
fn seeded(strategy: Strategy, seed: u64) -> Config {
    Config::default()
        .with_strategy(strategy)
        .with_time_limit(0)
        .with_population_size(8)
        .with_max_generations(6)
        .with_seed(seed)
}

#[test]
fn the_same_seed_gives_the_same_packing() {
    let (pieces, boundary) = mixed();
    for strategy in SEARCHES {
        let first = Packer3D::new(seeded(strategy, 11))
            .solve(&pieces, &boundary)
            .expect("solve");
        for run in 1..RUNS {
            let again = Packer3D::new(seeded(strategy, 11))
                .solve(&pieces, &boundary)
                .expect("solve");
            assert_eq!(
                layout(&again),
                layout(&first),
                "{strategy:?}: run {run} differs from run 0 with the same seed"
            );
            assert_eq!(again.utilization.to_bits(), first.utilization.to_bits());
        }
    }
}

/// The seed is what the search draws from: started from random genes rather
/// than the layer fill, different seeds explore different packings.
#[test]
fn different_seeds_search_differently() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use u_nesting_core::brkga::BrkgaConfig;
    use u_nesting_core::ga::GaConfig;
    use u_nesting_core::sa::SaConfig;
    use u_nesting_d3::{brkga_packing, ga_packing, sa_packing};

    let (pieces, boundary) = mixed();
    let run = |strategy: Strategy, seed: u64| {
        let config = seeded(strategy, seed);
        let flag = Arc::new(AtomicBool::new(false));
        let result = match strategy {
            Strategy::GeneticAlgorithm => ga_packing::run_ga_packing(
                &pieces,
                &boundary,
                &config,
                GaConfig::default()
                    .with_population_size(4)
                    .with_max_generations(1),
                flag,
                None,
                None,
            ),
            Strategy::Brkga => brkga_packing::run_brkga_packing(
                &pieces,
                &boundary,
                &config,
                BrkgaConfig::default()
                    .with_population_size(4)
                    .with_max_generations(1),
                flag,
                None,
                None,
            ),
            _ => sa_packing::run_sa_packing(
                &pieces,
                &boundary,
                &config,
                SaConfig::default().with_max_iterations(5),
                flag,
                None,
                None,
            ),
        };
        layout(&result.expect("valid settings"))
    };
    for strategy in SEARCHES {
        let layouts: std::collections::HashSet<Vec<Key>> =
            (0..8).map(|seed| run(strategy, seed)).collect();
        assert!(
            layouts.len() > 1,
            "{strategy:?}: eight seeds gave one packing"
        );
        // And each of them again gives its own.
        assert_eq!(run(strategy, 5), run(strategy, 5), "{strategy:?}");
    }
}

/// Packings the simple layer fill completes, which a search on a tiny budget
/// (four individuals, two generations) used to leave short -- one box of two
/// in up to 34 of 40 runs, and with gravity and stability two of three.
fn easy_cases() -> Vec<(&'static str, Vec<Geometry3D>, Boundary3D)> {
    let any = OrientationConstraint::Any;
    vec![
        (
            "two slabs that fit lying flat",
            vec![Geometry3D::new("slab", 40.0, 20.0, 10.0)
                .with_quantity(2)
                .with_orientation(any)],
            Boundary3D::new(50.0, 50.0, 12.0),
        ),
        (
            "two posts that fit standing",
            vec![Geometry3D::new("post", 10.0, 10.0, 40.0)
                .with_quantity(2)
                .with_orientation(any)],
            Boundary3D::new(25.0, 12.0, 45.0),
        ),
        (
            "three crates under gravity and stability",
            vec![Geometry3D::new("crate", 20.0, 20.0, 10.0)
                .with_quantity(3)
                .with_orientation(any)],
            Boundary3D::new(40.0, 40.0, 30.0)
                .with_gravity(true)
                .with_stability(true),
        ),
        (
            "a stack of three under gravity and stability",
            vec![
                Geometry3D::new("base", 40.0, 40.0, 10.0).with_orientation(any),
                Geometry3D::new("mid", 30.0, 30.0, 10.0).with_orientation(any),
                Geometry3D::new("top", 20.0, 20.0, 10.0).with_orientation(any),
            ],
            Boundary3D::new(40.0, 40.0, 40.0)
                .with_gravity(true)
                .with_stability(true),
        ),
    ]
}

#[test]
fn no_search_packs_less_than_the_simple_fill() {
    for (name, pieces, boundary) in easy_cases() {
        let total: usize = pieces.iter().map(u_nesting_core::Geometry::quantity).sum();
        let fill = Packer3D::new(Config::default().with_strategy(Strategy::BottomLeftFill))
            .solve(&pieces, &boundary)
            .expect("solve");
        assert_eq!(fill.placements.len(), total, "{name}: the fill itself");
        for strategy in SEARCHES {
            for seed in 0..40 {
                let config = Config::default()
                    .with_strategy(strategy)
                    .with_population_size(4)
                    .with_max_generations(2)
                    .with_seed(seed);
                let r = Packer3D::new(config)
                    .solve(&pieces, &boundary)
                    .expect("solve");
                assert_eq!(
                    r.placements.len(),
                    total,
                    "{strategy:?} seed {seed}, {name}: {r:?}"
                );
                assert!(r.utilization >= fill.utilization - 1e-9, "{name}");
                assert!(r.unplaced.is_empty(), "{name}");
            }
        }
    }
}
