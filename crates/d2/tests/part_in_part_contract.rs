//! Part-in-part, fuzzed against the independent layout checker.
//!
//! Random frames (a rectangle or an octagon with a rectangular hole, a notched hole, or
//! two holes, allowing part-in-part) and random chips go onto a random sheet through every strategy that
//! places by no-fit polygons. Whatever the layout is -- chips in holes, chips beside
//! the frames, pieces left out -- [`u_nesting_d2::check::check_layout`], which shares
//! no placement code with the nesting and judges a piece in a hole by the hole, must
//! find nothing wrong. A panic or a refusal of a valid request also fails.

use proptest::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use u_nesting_d2::check::{check_layout, LayoutRules};
use u_nesting_d2::{
    Boundary2D, Config, Geometry2D, Nester2D, SolveResult, Solver, Strategy as NestStrategy,
};

/// The holes of a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Holes {
    /// One rectangular hole.
    One,
    /// One rectangular hole with a bite taken out of its right side (a C opening rightwards).
    Notched,
    /// Two rectangular holes side by side, a rib of 2 between them.
    Two,
}

/// A frame `w` x `h` whose holes are inset by `inset` on every side; with `octagon` the
/// outline has its four corners cut off (by half the inset, so the holes stay inside).
fn frame(id: &str, w: f64, h: f64, inset: f64, holes: Holes, octagon: bool) -> Geometry2D {
    let (x0, y0, x1, y1) = (inset, inset, w - inset, h - inset);
    let outline = if octagon {
        let c = inset / 2.0;
        vec![
            (c, 0.0),
            (w - c, 0.0),
            (w, c),
            (w, h - c),
            (w - c, h),
            (c, h),
            (0.0, h - c),
            (0.0, c),
        ]
    } else {
        vec![(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    };
    let geometry = Geometry2D::new(id).with_polygon(outline);
    let geometry = match holes {
        Holes::One => geometry.with_hole(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
        Holes::Notched => {
            let (my, depth) = ((y0 + y1) / 2.0, (x1 - x0) / 2.0);
            geometry.with_hole(vec![
                (x0, y0),
                (x1, y0),
                (x1, my - 1.0),
                (x1 - depth, my - 1.0),
                (x1 - depth, my + 1.0),
                (x1, my + 1.0),
                (x1, y1),
                (x0, y1),
            ])
        }
        Holes::Two => {
            let mid = (x0 + x1) / 2.0;
            geometry
                .with_hole(vec![(x0, y0), (mid - 1.0, y0), (mid - 1.0, y1), (x0, y1)])
                .with_hole(vec![(mid + 1.0, y0), (x1, y0), (x1, y1), (mid + 1.0, y1)])
        }
    };
    geometry.with_part_in_part(true)
}

fn l_chip(id: &str, a: f64) -> Geometry2D {
    Geometry2D::new(id).with_polygon(vec![
        (0.0, 0.0),
        (a, 0.0),
        (a, a / 2.0),
        (a / 2.0, a / 2.0),
        (a / 2.0, a),
        (0.0, a),
    ])
}

#[derive(Debug, Clone)]
struct Case {
    frames: Vec<(f64, f64, f64, Holes, bool)>,
    chips: Vec<(f64, f64, bool, bool)>,
    sheet: (f64, f64),
    spacing: f64,
    margin: f64,
}

fn case() -> impl proptest::strategy::Strategy<Value = Case> {
    let holes = prop::sample::select(vec![Holes::One, Holes::Notched, Holes::Two]);
    let frames = prop::collection::vec(
        (
            22.0..42.0_f64,
            22.0..42.0_f64,
            4.0..8.0_f64,
            holes,
            any::<bool>(),
        ),
        1..3,
    );
    let chips = prop::collection::vec(
        (3.0..9.0_f64, 3.0..9.0_f64, any::<bool>(), any::<bool>()),
        2..9,
    );
    case_of(frames, chips)
}

/// One large frame with two holes, each wide enough for several chips: the layouts that
/// have to use both holes of the same piece.
fn two_hole_case() -> impl proptest::strategy::Strategy<Value = Case> {
    let frames = prop::collection::vec(
        (
            36.0..44.0_f64,
            24.0..34.0_f64,
            4.0..5.0_f64,
            Just(Holes::Two),
            any::<bool>(),
        ),
        1..2,
    );
    // many chips of about a third of a hole, so one hole fills up and the next is needed
    let chips = prop::collection::vec(
        (6.0..11.0_f64, 6.0..11.0_f64, any::<bool>(), any::<bool>()),
        7..13,
    );
    case_of(frames, chips)
}

fn case_of(
    frames: impl proptest::strategy::Strategy<Value = Vec<(f64, f64, f64, Holes, bool)>>,
    chips: impl proptest::strategy::Strategy<Value = Vec<(f64, f64, bool, bool)>>,
) -> impl proptest::strategy::Strategy<Value = Case> {
    (
        frames,
        chips,
        (50.0..120.0_f64, 40.0..80.0_f64),
        prop::sample::select(vec![0.0_f64, 0.5, 1.0, 2.0]),
        prop::sample::select(vec![0.0_f64, 0.5, 1.0]),
    )
        .prop_map(|(frames, chips, sheet, spacing, margin)| Case {
            frames,
            chips,
            sheet,
            spacing,
            margin,
        })
}

fn geometries(case: &Case) -> Vec<Geometry2D> {
    let mut all: Vec<Geometry2D> = case
        .frames
        .iter()
        .enumerate()
        .map(|(i, &(w, h, inset, holes, octagon))| {
            frame(&format!("frame{i}"), w, h, inset, holes, octagon)
        })
        .collect();
    for (i, &(a, b, l_shaped, rotate)) in case.chips.iter().enumerate() {
        let id = format!("chip{i}");
        let chip = if l_shaped {
            l_chip(&id, a.max(b))
        } else {
            Geometry2D::rectangle(&id, a, b)
        };
        all.push(if rotate {
            chip.with_rotations_deg(vec![0.0, 90.0])
        } else {
            chip
        });
    }
    all
}

/// The two coverage tests read process-wide counters that only they write; they take turns.
static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// How many layouts so far put a chip inside a frame's outline -- in its hole, since
/// the frame's material is never shared.
static CHIPS_IN_HOLES: AtomicUsize = AtomicUsize::new(0);

/// Whether some placed chip's reference point lies inside a placed (unrotated) frame's box.
fn a_chip_is_in_a_frame(case: &Case, result: &SolveResult<f64>) -> bool {
    let frame_boxes: Vec<(f64, f64, f64, f64)> = result
        .placements
        .iter()
        .filter_map(|p| {
            let i: usize = p.geometry_id.as_str().strip_prefix("frame")?.parse().ok()?;
            let (w, h, _, _, _) = case.frames[i];
            Some((
                p.position[0],
                p.position[1],
                p.position[0] + w,
                p.position[1] + h,
            ))
        })
        .collect();
    result
        .placements
        .iter()
        .filter(|p| p.geometry_id.as_str().starts_with("chip"))
        .any(|p| {
            frame_boxes.iter().any(|&(x0, y0, x1, y1)| {
                p.position[0] > x0 && p.position[0] < x1 && p.position[1] > y0 && p.position[1] < y1
            })
        })
}

/// Layouts that put a chip inside an octagonal frame's outline.
static CHIPS_IN_OCTAGONS: AtomicUsize = AtomicUsize::new(0);
/// Layouts that put chips in both holes of one two-hole frame.
static BOTH_HOLES_USED: AtomicUsize = AtomicUsize::new(0);

/// Tallies how the layout used the new frame shapes. A chip counts for a frame when its
/// reference point lies inside the placed frame's box (octagon), or inside a hole's span
/// (two-hole frame -- the holes are the left and right halves around the rib).
fn tally(case: &Case, result: &SolveResult<f64>) {
    if a_chip_is_in_a_frame(case, result) {
        CHIPS_IN_HOLES.fetch_add(1, Ordering::Relaxed);
    }
    let chips: Vec<(f64, f64)> = result
        .placements
        .iter()
        .filter(|p| p.geometry_id.as_str().starts_with("chip"))
        .map(|p| (p.position[0], p.position[1]))
        .collect();
    let (mut in_octagon, mut both_holes) = (false, false);
    for p in &result.placements {
        let Some(i) = p
            .geometry_id
            .as_str()
            .strip_prefix("frame")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let (w, h, inset, holes, octagon) = case.frames[i];
        let (fx, fy) = (p.position[0], p.position[1]);
        let inside = |&(x, y): &(f64, f64)| x > fx && x < fx + w && y > fy && y < fy + h;
        if octagon && chips.iter().any(inside) {
            in_octagon = true;
        }
        if holes == Holes::Two {
            let mid = fx + w / 2.0;
            let in_hole = |left: bool| {
                chips.iter().any(|&(x, y)| {
                    y > fy + inset
                        && y < fy + h - inset
                        && if left {
                            x > fx + inset && x < mid - 1.0
                        } else {
                            x > mid + 1.0 && x < fx + w - inset
                        }
                })
            };
            if in_hole(true) && in_hole(false) {
                both_holes = true;
            }
        }
    }
    if in_octagon {
        CHIPS_IN_OCTAGONS.fetch_add(1, Ordering::Relaxed);
    }
    if both_holes {
        BOTH_HOLES_USED.fetch_add(1, Ordering::Relaxed);
    }
}

fn solve_and_check(case: &Case, config: Config) -> Result<SolveResult<f64>, TestCaseError> {
    let geoms = geometries(case);
    let boundary = Boundary2D::rectangle(case.sheet.0, case.sheet.1);
    let rules = LayoutRules {
        spacing: case.spacing,
        margin: case.margin,
        multi_sheet: false,
    };
    let result = Nester2D::new(
        config
            .with_spacing(case.spacing)
            .with_margin(case.margin)
            .with_time_limit(2_000),
    )
    .solve(&geoms, &boundary)
    .map_err(|e| TestCaseError::fail(format!("a valid request was refused: {e:?} for {case:?}")))?;
    let violations = check_layout(
        &geoms,
        &boundary,
        &rules,
        &result.placements,
        &result.unplaced,
    );
    prop_assert!(
        violations.is_empty(),
        "{violations:?} for {case:?} -> {:?}",
        result.placements
    );
    Ok(result)
}

/// The NFP strategy places in the order given: frames first, chips after. The run also
/// has to have put chips in holes in a good share of the layouts, or it fuzzes nothing.
#[test]
fn the_nfp_strategy_never_breaks_the_checker_and_uses_the_holes() {
    let _turn = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    CHIPS_IN_HOLES.store(0, Ordering::Relaxed);
    CHIPS_IN_OCTAGONS.store(0, Ordering::Relaxed);
    BOTH_HOLES_USED.store(0, Ordering::Relaxed);
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(60));
    runner
        .run(&case(), |case| {
            let result = solve_and_check(
                &case,
                Config::default().with_strategy(NestStrategy::NfpGuided),
            )?;
            tally(&case, &result);
            Ok(())
        })
        .expect("every layout passes the checker");
    let in_holes = CHIPS_IN_HOLES.load(Ordering::Relaxed);
    assert!(
        in_holes >= 15,
        "only {in_holes} of 60 layouts put a chip in a hole: the fuzz does not exercise the pockets"
    );
    let in_octagons = CHIPS_IN_OCTAGONS.load(Ordering::Relaxed);
    assert!(
        in_octagons >= 5,
        "only {in_octagons} of 60 layouts put a chip in an octagonal frame: the outline shape is not exercised"
    );
}

/// Frames with two holes: the pockets of one piece are packed independently, and a chip
/// in one hole must not be judged against the other hole's rim. Both holes have to be
/// used in a fair share of the layouts, or the case fuzzes nothing new.
#[test]
fn the_nfp_strategy_fills_both_holes_of_a_frame() {
    let _turn = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    BOTH_HOLES_USED.store(0, Ordering::Relaxed);
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(40));
    runner
        .run(&two_hole_case(), |case| {
            let result = solve_and_check(
                &case,
                Config::default().with_strategy(NestStrategy::NfpGuided),
            )?;
            tally(&case, &result);
            Ok(())
        })
        .expect("every layout passes the checker");
    let both = BOTH_HOLES_USED.load(Ordering::Relaxed);
    assert!(
        both >= 10,
        "only {both} of 40 layouts used both holes of the frame: the fuzz does not exercise two holes"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn the_genetic_algorithm_never_breaks_the_checker(case in case()) {
        solve_and_check(
            &case,
            Config::default()
                .with_strategy(NestStrategy::GeneticAlgorithm)
                .with_population_size(6)
                .with_max_generations(4)
                .with_seed(7),
        )?;
    }

    #[test]
    fn alns_never_breaks_the_checker(case in case()) {
        solve_and_check(
            &case,
            Config::default()
                .with_strategy(NestStrategy::Alns)
                .with_max_iterations(40)
                .with_seed(7),
        )?;
    }

    #[test]
    fn the_other_searches_never_break_the_checker(case in case()) {
        for strategy in [
            NestStrategy::Brkga,
            NestStrategy::SimulatedAnnealing,
            NestStrategy::Gdrr,
        ] {
            let mut config = Config::default().with_strategy(strategy).with_seed(7);
            config = match strategy {
                NestStrategy::Brkga => config.with_population_size(6).with_max_generations(4),
                _ => config.with_max_iterations(40),
            };
            solve_and_check(&case, config)?;
        }
    }
}
