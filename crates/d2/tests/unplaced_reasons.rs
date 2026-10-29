//! A piece left unplaced says why: it can never fit a sheet of this size
//! (`exceeds_boundary` -- more sheets will not help) or the layout ran out of
//! room (`no_room`).

use u_nesting_d2::{Boundary2D, Config, Geometry2D, Nester2D, Solver, Strategy, UnplacedReason};

fn reasons(result: &u_nesting_d2::SolveResult<f64>) -> Vec<(&str, UnplacedReason)> {
    result
        .unplaced_reasons
        .iter()
        .map(|(id, reason)| (id.as_str(), *reason))
        .collect()
}

#[test]
fn a_piece_larger_than_the_sheet_exceeds_the_boundary_and_a_full_sheet_has_no_room() {
    let pieces = [
        Geometry2D::rectangle("big", 500.0, 500.0),
        Geometry2D::rectangle("tile", 40.0, 40.0).with_quantity(8),
    ];
    let boundary = Boundary2D::rectangle(100.0, 100.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::NfpGuided] {
        let result = Nester2D::new(Config::default().with_strategy(strategy))
            .solve(&pieces, &boundary)
            .expect("solved");
        assert_eq!(
            result.unplaced.len(),
            result.unplaced_reasons.len(),
            "{strategy:?}"
        );
        let by_id = reasons(&result);
        assert!(
            by_id.contains(&("big", UnplacedReason::ExceedsBoundary)),
            "{strategy:?}: {by_id:?}"
        );
        // 8 tiles of 40 x 40 cannot all go on one 100 x 100 sheet: the ones
        // that remain fit a sheet and found no room on this one.
        assert!(
            by_id.contains(&("tile", UnplacedReason::NoRoom)),
            "{strategy:?}: {by_id:?}"
        );
    }
}

#[test]
fn the_margin_counts_and_a_rotation_that_fits_saves_the_piece() {
    let boundary = Boundary2D::rectangle(100.0, 50.0);
    // 45 x 90 fits only turned a quarter; with a margin of 5 the sheet's room
    // is 90 x 40, so the turned piece (90 x 45) still does not fit.
    let piece = || Geometry2D::rectangle("bar", 45.0, 90.0).with_rotations_deg(vec![0.0, 90.0]);
    let free = Nester2D::new(Config::default())
        .solve(&[piece()], &boundary)
        .expect("solved");
    assert!(
        free.unplaced.is_empty(),
        "a quarter turn fits: {:?}",
        free.unplaced
    );

    let margined = Nester2D::new(Config::default().with_margin(5.0))
        .solve(&[piece()], &boundary)
        .expect("solved");
    assert_eq!(
        reasons(&margined),
        vec![("bar", UnplacedReason::ExceedsBoundary)]
    );
}

#[test]
fn a_multi_sheet_solve_classifies_too() {
    let pieces = [
        Geometry2D::rectangle("big", 500.0, 500.0),
        Geometry2D::rectangle("tile", 40.0, 40.0).with_quantity(3),
    ];
    let boundary = Boundary2D::rectangle(100.0, 100.0);
    let result = Nester2D::new(Config::default())
        .solve_multi_strip(&pieces, &boundary)
        .expect("solved");
    assert_eq!(
        reasons(&result),
        vec![("big", UnplacedReason::ExceedsBoundary)]
    );
    assert!(result.placements.len() >= 3, "the tiles go on more sheets");
}
