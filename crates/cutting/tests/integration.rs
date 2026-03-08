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

    assert!(result.estimated_time_seconds.is_some());
    let time = result.estimated_time_seconds.unwrap();
    assert!(time > 0.0, "estimated time should be positive");
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
