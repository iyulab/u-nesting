//! Layouts use as little of a boundary's length as they can, whichever way the
//! boundary is oriented: a wide sheet is filled in columns and a tall one in
//! rows.

use u_nesting_d2::{Boundary2D, Config, Geometry2D, Nester2D, SolveResult, Solver, Strategy};

fn l_pieces() -> Vec<Geometry2D> {
    (0..8)
        .map(|i| {
            let (w, h, t) = (30.0 + 6.0 * i as f64, 25.0 + 4.0 * i as f64, 10.0);
            Geometry2D::new(format!("L{i}"))
                .with_polygon(vec![(0.0, 0.0), (w, 0.0), (w, t), (t, t), (t, h), (0.0, h)])
                .with_quantity(5)
                .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0])
        })
        .collect()
}

/// Length used: the extent along the boundary's longer side.
fn length(result: &SolveResult<f64>, wide: bool) -> f64 {
    result.used_bounding_box[if wide { 0 } else { 1 }]
}

fn solve(strategy: Strategy, boundary: &Boundary2D) -> SolveResult<f64> {
    Nester2D::new(Config::default().with_strategy(strategy).with_time_limit(0))
        .solve(&l_pieces(), boundary)
        .unwrap()
}

#[test]
fn the_same_parts_take_a_similar_length_on_a_wide_and_a_tall_sheet() {
    let tall = Boundary2D::rectangle(500.0, 5000.0);
    let wide = Boundary2D::rectangle(5000.0, 500.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::NfpGuided] {
        let (t, w) = (solve(strategy, &tall), solve(strategy, &wide));
        assert_eq!(
            (t.placements.len(), w.placements.len()),
            (40, 40),
            "{strategy:?}"
        );
        let (lt, lw) = (length(&t, false), length(&w, true));
        assert!(
            lt.max(lw) <= 1.5 * lt.min(lw),
            "{strategy:?}: {lt} along a tall sheet but {lw} along a wide one"
        );
    }
}

/// Placement by no-fit polygons searches shapes, bounding-box rows do not; on
/// either orientation it must not need more length than the row packer.
#[test]
fn shape_aware_placement_is_no_longer_than_bounding_box_rows() {
    for (boundary, wide) in [
        (Boundary2D::rectangle(500.0, 5000.0), false),
        (Boundary2D::rectangle(5000.0, 500.0), true),
    ] {
        let rows = length(&solve(Strategy::BottomLeftFill, &boundary), wide);
        let nfp = length(&solve(Strategy::NfpGuided, &boundary), wide);
        assert!(
            nfp <= rows,
            "wide={wide}: NFP placement {nfp} vs rows {rows}"
        );
    }
}
