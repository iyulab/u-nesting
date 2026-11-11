//! Part-in-part, measured: frames with a hole that takes small chips, on a sheet the
//! frames fill -- every piece is placed only if the chips go in the holes.
//!
//! The searches rank a layout by how many pieces it places before how much area it
//! covers, so this is the instance where holes change their answer: with the holes
//! kept empty the best layout drops pieces, with `allow_part_in_part` it places all.
//!
//! `cargo run --release -p u-nesting-d2 --example part_in_part`
//!
//! For each strategy that places by no-fit polygons it solves the same pieces with
//! the frames' holes kept empty and with `allow_part_in_part` set, and prints how
//! many chips were placed, the sheet utilization and the time. Every layout is also
//! put through [`u_nesting_d2::check::check_layout`], the checker that shares no
//! placement code with the nesting.
//!
//! The search budgets can be raised with the environment variables `PIP_TIME_MS`,
//! `PIP_POPULATION`, `PIP_GENERATIONS` and `PIP_ITERATIONS`.

use u_nesting_core::solver::{Config, Solver, Strategy};
use u_nesting_d2::check::{check_layout, LayoutRules};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};

/// A square frame with a square hole in its middle.
fn frame(side: f64, hole: f64, allow: bool) -> Geometry2D {
    let m = (side - hole) / 2.0;
    Geometry2D::rectangle("frame", side, side)
        .with_hole(vec![
            (m, m),
            (m + hole, m),
            (m + hole, m + hole),
            (m, m + hole),
        ])
        .with_quantity(3)
        .with_part_in_part(allow)
}

/// A budget from the environment, or `default`.
fn budget(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Three 30 x 30 frames (20 x 20 holes) side by side with the spacing between them
    // fill a 93 x 30 sheet; the twelve chips of 8 x 8 fit only in the holes (a 2 x 2
    // block of chips and the spacing is 17 x 17 of a hole's 18 x 18 left by the spacing).
    let boundary = Boundary2D::rectangle(93.0, 30.0);
    let spacing = 1.0;
    let rules = LayoutRules {
        spacing,
        margin: 0.0,
        multi_sheet: false,
    };
    let strategies = [
        (Strategy::NfpGuided, vec![0u64]),
        (Strategy::GeneticAlgorithm, vec![1, 2, 3]),
        (Strategy::Alns, vec![1, 2, 3]),
    ];

    println!("strategy | holes | seed | pieces placed (of 15) | utilization | check | ms");
    println!("---|---|---|---|---|---|---");
    for (strategy, seeds) in strategies {
        for allow in [false, true] {
            for &seed in &seeds {
                let geometries = vec![
                    frame(30.0, 20.0, allow),
                    Geometry2D::rectangle("chip", 8.0, 8.0).with_quantity(12),
                ];
                let mut config = Config::default()
                    .with_strategy(strategy)
                    .with_spacing(spacing)
                    .with_time_limit(budget("PIP_TIME_MS", 8_000));
                if strategy.reads_setting(u_nesting_core::solver::Dimension::TwoD, "seed") {
                    config = config.with_seed(seed);
                }
                if strategy
                    .reads_setting(u_nesting_core::solver::Dimension::TwoD, "population_size")
                {
                    config = config
                        .with_population_size(budget("PIP_POPULATION", 20) as usize)
                        .with_max_generations(budget("PIP_GENERATIONS", 40) as u32);
                }
                if strategy.reads_setting(u_nesting_core::solver::Dimension::TwoD, "max_iterations")
                {
                    config = config.with_max_iterations(budget("PIP_ITERATIONS", 300));
                }
                let result = Nester2D::new(config)
                    .solve(&geometries, &boundary)
                    .expect("solved");
                let placed = result.placements.len();
                let violations = check_layout(
                    &geometries,
                    &boundary,
                    &rules,
                    &result.placements,
                    &result.unplaced,
                );
                println!(
                    "{strategy:?} | {} | {seed} | {placed} | {:.1} % | {} | {}",
                    if allow { "open" } else { "empty" },
                    result.utilization * 100.0,
                    if violations.is_empty() {
                        "0 violations".to_string()
                    } else {
                        format!("{} VIOLATIONS", violations.len())
                    },
                    result.computation_time_ms,
                );
            }
        }
    }

    scale(&rules);
}

/// The cost of the pockets: forty frames, each with a notched (C-shaped) hole, and two
/// hundred chips on a large sheet, placed by `nfp`. The notch gives every pocket a bay.
fn scale(rules: &LayoutRules) {
    let c_frame = |allow: bool| {
        Geometry2D::rectangle("frame", 30.0, 30.0)
            .with_hole(vec![
                (5.0, 5.0),
                (25.0, 5.0),
                (25.0, 13.0),
                (13.0, 13.0),
                (13.0, 17.0),
                (25.0, 17.0),
                (25.0, 25.0),
                (5.0, 25.0),
            ])
            .with_quantity(40)
            .with_part_in_part(allow)
    };
    let boundary = Boundary2D::rectangle(400.0, 150.0);
    println!();
    println!("scale: 40 notched frames + 200 chips of 6 x 6, 400 x 150 sheet, nfp");
    println!("holes | pieces placed (of 240) | utilization | check | ms");
    println!("---|---|---|---|---");
    for allow in [false, true] {
        let geometries = vec![
            c_frame(allow),
            Geometry2D::rectangle("chip", 6.0, 6.0).with_quantity(200),
        ];
        let config = Config::default()
            .with_strategy(Strategy::NfpGuided)
            .with_spacing(rules.spacing)
            .with_time_limit(120_000);
        let result = Nester2D::new(config)
            .solve(&geometries, &boundary)
            .expect("solved");
        let violations = check_layout(
            &geometries,
            &boundary,
            rules,
            &result.placements,
            &result.unplaced,
        );
        println!(
            "{} | {} | {:.1} % | {} | {}",
            if allow { "open" } else { "empty" },
            result.placements.len(),
            result.utilization * 100.0,
            if violations.is_empty() {
                "0 violations".to_string()
            } else {
                format!("{} VIOLATIONS", violations.len())
            },
            result.computation_time_ms,
        );
    }
}
