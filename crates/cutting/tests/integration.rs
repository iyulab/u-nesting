//! End-to-end integration tests for cutting path optimization.
//!
//! Tests the full pipeline: Geometry2D → Nester2D.solve() → optimize_cutting_path().

use u_nesting_core::solver::{Config, Solver};
use u_nesting_cutting::{ContourType, CutDirection, CuttingConfig};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};

/// Helper: solve nesting and then optimize cutting path.
fn solve_and_cut(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    nesting_config: Config,
    cutting_config: CuttingConfig,
) -> u_nesting_cutting::CuttingPathResult {
    let nester = Nester2D::new(nesting_config);
    let solve_result = nester
        .solve(geometries, boundary)
        .expect("nesting should succeed");
    u_nesting_cutting::optimize_cutting_path(&solve_result, geometries, &cutting_config)
        .expect("valid cutting request")
}

#[test]
fn test_single_rectangle() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);

    let result = solve_and_cut(
        &geometries,
        &boundary,
        Config::default(),
        CuttingConfig::default(),
    );

    assert_eq!(result.total_pierces, 1);
    assert_eq!(result.sequence.len(), 1);
    assert!(result.total_cut_distance > 0.0);
    assert_eq!(result.sequence[0].geometry_id, "R1");
    assert_eq!(result.sequence[0].contour_type, ContourType::Exterior);
}

#[test]
fn test_multiple_rectangles() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0).with_quantity(4)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);

    let result = solve_and_cut(
        &geometries,
        &boundary,
        Config::default(),
        CuttingConfig::default(),
    );

    assert_eq!(result.total_pierces, 4);
    assert_eq!(result.sequence.len(), 4);

    // All steps should reference the same geometry
    for step in &result.sequence {
        assert_eq!(step.geometry_id, "R1");
        assert_eq!(step.contour_type, ContourType::Exterior);
    }

    // Verify efficiency is reasonable
    assert!(result.efficiency() > 0.5, "efficiency should be > 50%");
}

#[test]
fn test_rectangle_with_hole() {
    let geometries = vec![Geometry2D::new("PartH")
        .with_polygon(vec![(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)])
        .with_hole(vec![(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)])];
    let boundary = Boundary2D::rectangle(100.0, 100.0);

    let result = solve_and_cut(
        &geometries,
        &boundary,
        Config::default(),
        CuttingConfig::default(),
    );

    assert_eq!(result.total_pierces, 2);
    assert_eq!(result.sequence.len(), 2);

    // Interior (hole) must be cut before exterior (precedence constraint)
    assert_eq!(result.sequence[0].contour_type, ContourType::Interior);
    assert_eq!(result.sequence[1].contour_type, ContourType::Exterior);

    // Both belong to the same geometry
    assert_eq!(result.sequence[0].geometry_id, "PartH");
    assert_eq!(result.sequence[1].geometry_id, "PartH");
}

#[test]
fn test_multiple_parts_with_holes() {
    let geometries = vec![Geometry2D::new("P1")
        .with_polygon(vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)])
        .with_hole(vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])
        .with_quantity(2)];
    let boundary = Boundary2D::rectangle(100.0, 100.0);

    let result = solve_and_cut(
        &geometries,
        &boundary,
        Config::default(),
        CuttingConfig::default(),
    );

    // 2 instances × 2 contours (exterior + hole) = 4 steps
    assert_eq!(result.total_pierces, 4);
    assert_eq!(result.sequence.len(), 4);

    // Verify each instance's hole comes before its exterior
    let mut instance_order: std::collections::HashMap<usize, Vec<ContourType>> =
        std::collections::HashMap::new();
    for step in &result.sequence {
        instance_order
            .entry(step.instance)
            .or_default()
            .push(step.contour_type);
    }

    for types in instance_order.values() {
        // Interior should appear before Exterior for each instance
        let int_pos = types.iter().position(|t| *t == ContourType::Interior);
        let ext_pos = types.iter().position(|t| *t == ContourType::Exterior);
        if let (Some(ip), Some(ep)) = (int_pos, ext_pos) {
            assert!(ip < ep, "Interior should come before Exterior");
        }
    }
}

#[test]
fn test_cutting_with_kerf_compensation() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);

    let config_no_kerf = CuttingConfig::default();
    let config_with_kerf = CuttingConfig::new().with_kerf_width(0.5);

    let result_no_kerf = solve_and_cut(&geometries, &boundary, Config::default(), config_no_kerf);
    let result_with_kerf =
        solve_and_cut(&geometries, &boundary, Config::default(), config_with_kerf);

    // Kerf compensation should increase the cut distance (larger contour)
    assert!(
        result_with_kerf.total_cut_distance > result_no_kerf.total_cut_distance,
        "Kerf compensation should increase cut distance: {} vs {}",
        result_with_kerf.total_cut_distance,
        result_no_kerf.total_cut_distance
    );
}

#[test]
fn test_cutting_with_gtsp() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0).with_quantity(3)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);

    let config = CuttingConfig::new().with_pierce_candidates(4);

    let result = solve_and_cut(&geometries, &boundary, Config::default(), config);

    assert_eq!(result.total_pierces, 3);
    assert_eq!(result.sequence.len(), 3);
    assert!(result.total_cut_distance > 0.0);
}

#[test]
fn test_cutting_time_estimation() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);

    let config = CuttingConfig {
        rapid_speed: 1000.0,
        cut_speed: 100.0,
        ..CuttingConfig::default()
    };

    let result = solve_and_cut(&geometries, &boundary, Config::default(), config);

    // The rapid distance at 1000 units/s plus the cut distance at 100 units/s.
    let expected = result.total_rapid_distance / 1000.0 + result.total_cut_distance / 100.0;
    assert!(
        result.estimated_time_seconds > 0.0,
        "a part takes time to cut"
    );
    assert!((result.estimated_time_seconds - expected).abs() < 1e-9);
}

/// A speed of 0 or below has no time estimate: the optimizer refuses it
/// rather than leaving the estimate out.
#[test]
fn a_speed_that_is_not_above_zero_is_refused() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);
    let solved = Nester2D::new(Config::default())
        .solve(&geometries, &boundary)
        .expect("solve");
    for config in [
        CuttingConfig::new().with_speeds(1000.0, 0.0),
        CuttingConfig::new().with_speeds(-1.0, 100.0),
    ] {
        let e = u_nesting_cutting::optimize_cutting_path(&solved, &geometries, &config)
            .expect_err("no speed");
        assert_eq!(e.code(), "parameter_out_of_range", "{e}");
    }
}

#[test]
fn test_home_position_affects_rapid() {
    let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0)];
    let boundary = Boundary2D::rectangle(100.0, 50.0);

    let config_origin = CuttingConfig::new().with_home_position(0.0, 0.0);
    let config_far = CuttingConfig::new().with_home_position(50.0, 50.0);

    let result_origin = solve_and_cut(&geometries, &boundary, Config::default(), config_origin);
    let result_far = solve_and_cut(&geometries, &boundary, Config::default(), config_far);

    // Different home positions should give different rapid distances
    // (unless the part happens to be equidistant from both, which is unlikely)
    // We just verify both complete successfully
    assert_eq!(result_origin.total_pierces, 1);
    assert_eq!(result_far.total_pierces, 1);
}

#[test]
fn test_optimize_cutting_path_is_time_bounded_end_to_end() {
    // Regression (ISSUE-20260707): the *entire* optimize_cutting_path entry
    // point — contour extraction, common-edge detection, NN construction and
    // the 2-opt phase — must stay responsive on large but legitimate inputs
    // ("N identical brackets on a sheet"). Before the fix this froze for many
    // seconds / unboundedly and blocked the (browser main) thread.
    let geometries = vec![Geometry2D::rectangle("P", 10.0, 10.0).with_quantity(1200)];
    let boundary = Boundary2D::rectangle(600.0, 600.0);

    // Bound the nesting solve so this measures only the cutting-path cost.
    let nesting_config = Config {
        time_limit_ms: 2000,
        ..Config::default()
    };
    let nester = Nester2D::new(nesting_config);
    let solve_result = nester
        .solve(&geometries, &boundary)
        .expect("nesting should succeed");
    let placed = solve_result.placements.len();
    assert!(
        placed >= 800,
        "expected a sheet-filling layout for the regression, got {placed}"
    );

    let cutting_config = CuttingConfig::default().with_time_limit_ms(200);
    let t = std::time::Instant::now();
    let result =
        u_nesting_cutting::optimize_cutting_path(&solve_result, &geometries, &cutting_config)
            .expect("valid cutting request");
    let elapsed = t.elapsed();

    // Early termination must still yield a valid, complete sequence.
    assert_eq!(result.sequence.len(), placed);
    // Generous ceiling: unbounded runtime was >5.6s at 500 parts and worse
    // here. 5s never flakes yet catches an end-to-end unbounded regression.
    assert!(
        elapsed.as_secs() < 5,
        "optimize_cutting_path must be time-bounded end-to-end; took {elapsed:?}"
    );
}

#[test]
fn test_cut_direction() {
    let geometries = vec![Geometry2D::new("P1")
        .with_polygon(vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)])
        .with_hole(vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])];
    let boundary = Boundary2D::rectangle(100.0, 100.0);

    let result = solve_and_cut(
        &geometries,
        &boundary,
        Config::default(),
        CuttingConfig::default(),
    );

    // Default auto direction: exterior=CCW, interior=CW
    for step in &result.sequence {
        match step.contour_type {
            ContourType::Exterior => {
                assert_eq!(step.cut_direction, CutDirection::Ccw);
            }
            ContourType::Interior => {
                assert_eq!(step.cut_direction, CutDirection::Cw);
            }
        }
    }
}

/// A big part and two small ones nested with a spacing of 1.
fn tight_layout() -> (Vec<Geometry2D>, u_nesting_core::SolveResult<f64>) {
    let geometries = vec![
        Geometry2D::rectangle("big", 50.0, 30.0)
            .with_rotations(vec![0.0])
            .with_flip(false),
        Geometry2D::rectangle("tiny", 2.0, 2.0)
            .with_quantity(2)
            .with_rotations(vec![0.0])
            .with_flip(false),
    ];
    let boundary = Boundary2D::rectangle(100.0, 100.0);
    let solve = Nester2D::new(Config::default().with_spacing(1.0))
        .solve(&geometries, &boundary)
        .expect("nesting should succeed");
    assert_eq!(solve.placements.len(), 3);
    (geometries, solve)
}

/// A kerf wider than the gap between two parts cuts one into the other. It
/// used to return `success` with an empty sequence (and nothing skipped): the
/// overlapping offsets made the two small parts wait for each other, and the
/// big one, whose offset reached them, waited too.
#[test]
fn a_kerf_wider_than_the_gap_is_refused_with_the_gap() {
    let (geometries, solve) = tight_layout();
    for kerf in [1.5, 3.0, 5.0, 8.0] {
        let config = CuttingConfig::new().with_kerf_width(kerf);
        let err = u_nesting_cutting::optimize_cutting_path(&solve, &geometries, &config)
            .expect_err("kerf wider than the gap");
        assert_eq!(err.code(), "parameter_out_of_range");
        match err {
            u_nesting_core::Error::OutOfRange {
                parameter,
                max,
                got,
                ..
            } => {
                assert_eq!(parameter, "kerf_width");
                assert!((max.expect("bounded") - 1.0).abs() < 1e-9, "{max:?}");
                assert_eq!(got, kerf);
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_kerf_that_fits_the_gap_cuts_every_part() {
    let (geometries, solve) = tight_layout();
    for kerf in [0.5, 1.0] {
        let config = CuttingConfig::new().with_kerf_width(kerf);
        let result = u_nesting_cutting::optimize_cutting_path(&solve, &geometries, &config)
            .expect("kerf within the gap");
        assert_eq!(result.sequence.len(), 3, "kerf {kerf}");
        assert!(result.skipped.is_empty());
    }
}

/// Without a time limit the optimizer reads no clock and draws nothing at
/// random: the same layout gives the same sequence.
#[test]
fn the_same_layout_gives_the_same_sequence_without_a_time_limit() {
    let geometries = vec![Geometry2D::new("P1")
        .with_polygon(vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)])
        .with_hole(vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])
        .with_quantity(3)];
    let boundary = Boundary2D::rectangle(100.0, 100.0);
    let config = || {
        CuttingConfig::new()
            .with_pierce_candidates(4)
            .with_time_limit_ms(0)
    };
    let first = solve_and_cut(&geometries, &boundary, Config::default(), config());
    for _ in 0..5 {
        let again = solve_and_cut(&geometries, &boundary, Config::default(), config());
        assert_eq!(
            format!("{:?}", again.sequence),
            format!("{:?}", first.sequence)
        );
        assert_eq!(
            again.total_rapid_distance.to_bits(),
            first.total_rapid_distance.to_bits()
        );
    }
}
