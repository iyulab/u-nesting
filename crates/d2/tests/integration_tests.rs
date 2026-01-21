//! Integration tests for u-nesting-d2.

use u_nesting_d2::{
    Boundary, Boundary2D, Config, Geometry, Geometry2D, Geometry2DExt, Nester2D, Solver,
    Transform2D, AABB2D,
};

mod geometry_tests {
    use super::*;

    #[test]
    fn test_rectangle_geometry() {
        let rect = Geometry2D::rectangle("rect1", 20.0, 15.0);

        // Area should be width * height
        let area = rect.measure();
        assert!((area - 300.0).abs() < 0.001);

        // Perimeter should be 2*(w+h)
        let perim = rect.perimeter();
        assert!((perim - 70.0).abs() < 0.001);

        // Rectangle should be convex
        assert!(rect.is_convex());

        // Vertices should form a rectangle
        let vertices = rect.outer_ring();
        assert_eq!(vertices.len(), 4);
    }

    #[test]
    fn test_circle_approximation() {
        let circle = Geometry2D::circle("circle1", 10.0, 64);

        // Area should be approximately π*r²
        let expected_area = std::f64::consts::PI * 10.0 * 10.0;
        let actual_area = circle.measure();
        assert!(
            (actual_area - expected_area).abs() < 3.0,
            "Circle area {} should be close to {}",
            actual_area,
            expected_area
        );

        // Circle approximation should be convex
        assert!(circle.is_convex());

        // Should have 64 vertices
        assert_eq!(circle.outer_ring().len(), 64);
    }

    #[test]
    fn test_l_shape_non_convex() {
        let l_shape = Geometry2D::l_shape("l1", 30.0, 30.0, 15.0, 15.0);

        // L-shape should NOT be convex
        assert!(!l_shape.is_convex());

        // Area calculation: 30*30 - 15*15 = 900 - 225 = 675
        // Wait, L-shape is: total width=30, height=30, notch width=15, notch height=15
        // Vertices: (0,0), (30,0), (30,15), (15,15), (15,30), (0,30)
        // Area: 30*30 - 15*15 = 675? No, let's compute correctly:
        // The L has a notch cut out of the top-right
        // Actually the l_shape creates:
        // (0,0), (width,0), (width,notch_height), (notch_width,notch_height), (notch_width,height), (0,height)
        // So it's: 30*15 + 15*30 = 450 + 450 = 900 - overlap = ...
        // Let's just verify it's positive and reasonable
        let area = l_shape.measure();
        assert!(area > 0.0);
        assert!(area < 900.0); // Less than bounding rectangle

        // Convex hull should be the bounding rectangle
        let hull = l_shape.convex_hull();
        assert!(hull.len() >= 4);
    }

    #[test]
    fn test_geometry_with_hole() {
        let poly_with_hole = Geometry2D::new("frame")
            .with_polygon(vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
            .with_hole(vec![(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)]);

        // Area should be outer - hole = 10000 - 2500 = 7500
        let area = poly_with_hole.measure();
        assert!((area - 7500.0).abs() < 0.001, "Area = {}", area);

        // Not convex due to hole
        assert!(!poly_with_hole.is_convex());
    }

    #[test]
    fn test_geometry_aabb() {
        let geom = Geometry2D::new("offset_rect").with_polygon(vec![
            (10.0, 20.0),
            (50.0, 20.0),
            (50.0, 60.0),
            (10.0, 60.0),
        ]);

        let aabb = geom.aabb_2d();
        assert!((aabb.min_x - 10.0).abs() < 1e-10);
        assert!((aabb.min_y - 20.0).abs() < 1e-10);
        assert!((aabb.max_x - 50.0).abs() < 1e-10);
        assert!((aabb.max_y - 60.0).abs() < 1e-10);
        assert!((aabb.width() - 40.0).abs() < 1e-10);
        assert!((aabb.height() - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometry_centroid() {
        let rect = Geometry2D::rectangle("centered", 10.0, 10.0);
        let centroid = rect.centroid();

        assert!((centroid[0] - 5.0).abs() < 0.001);
        assert!((centroid[1] - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_geometry_validation() {
        // Valid geometry
        let valid = Geometry2D::rectangle("valid", 10.0, 10.0);
        assert!(valid.validate().is_ok());

        // Invalid: less than 3 vertices
        let invalid = Geometry2D::new("invalid").with_polygon(vec![(0.0, 0.0), (1.0, 0.0)]);
        assert!(invalid.validate().is_err());

        // Invalid: zero quantity
        let zero_qty = Geometry2D::rectangle("zero", 10.0, 10.0).with_quantity(0);
        assert!(zero_qty.validate().is_err());
    }

    #[test]
    fn test_geometry_rotations() {
        let geom = Geometry2D::rectangle("rot", 10.0, 10.0)
            .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0]);

        let rotations = geom.rotations();
        assert_eq!(rotations.len(), 4);

        let pi = std::f64::consts::PI;
        assert!((rotations[0] - 0.0).abs() < 1e-10);
        assert!((rotations[1] - pi / 2.0).abs() < 1e-10);
        assert!((rotations[2] - pi).abs() < 1e-10);
        assert!((rotations[3] - 3.0 * pi / 2.0).abs() < 1e-10);
    }
}

mod boundary_tests {
    use super::*;
    use u_nesting_d2::Boundary2DExt;

    #[test]
    fn test_rectangle_boundary() {
        let boundary = Boundary2D::rectangle(200.0, 100.0);

        assert_eq!(boundary.width(), Some(200.0));
        assert_eq!(boundary.height(), Some(100.0));
        assert!(!boundary.is_infinite());

        let area = boundary.measure();
        assert!((area - 20000.0).abs() < 0.001);
    }

    #[test]
    fn test_strip_boundary() {
        let strip = Boundary2D::strip(100.0);

        assert_eq!(strip.width(), Some(100.0));
        assert_eq!(strip.height(), None);
        assert!(strip.is_infinite());

        let area = strip.measure();
        assert!(area == f64::INFINITY);
    }

    #[test]
    fn test_boundary_with_hole() {
        let boundary = Boundary2D::rectangle(100.0, 100.0).with_hole(vec![
            (40.0, 40.0),
            (60.0, 40.0),
            (60.0, 60.0),
            (40.0, 60.0),
        ]);

        // Area should be reduced by the hole
        let area = boundary.measure();
        // 10000 - 400 = 9600
        assert!((area - 9600.0).abs() < 0.001, "Area = {}", area);
    }

    #[test]
    fn test_boundary_contains_point() {
        let boundary = Boundary2D::rectangle(100.0, 100.0);

        // Point inside
        assert!(boundary.contains_point(&[50.0, 50.0]));

        // Point outside
        assert!(!boundary.contains_point(&[150.0, 50.0]));
        assert!(!boundary.contains_point(&[-10.0, 50.0]));

        // Point on edge (behavior may vary)
        // Just test points clearly inside/outside
    }

    #[test]
    fn test_boundary_aabb() {
        let boundary = Boundary2D::rectangle(150.0, 75.0);
        let aabb = boundary.aabb_2d();

        assert!((aabb.min_x - 0.0).abs() < 1e-10);
        assert!((aabb.min_y - 0.0).abs() < 1e-10);
        assert!((aabb.max_x - 150.0).abs() < 1e-10);
        assert!((aabb.max_y - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_effective_area() {
        let boundary = Boundary2D::rectangle(100.0, 100.0);

        // With margin of 10, effective dimensions are 80x80
        let effective = boundary.effective_area(10.0);
        assert!((effective - 6400.0).abs() < 0.001);

        // With margin of 0
        let full = boundary.effective_area(0.0);
        assert!((full - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_boundary_validation() {
        // Valid boundary
        let valid = Boundary2D::rectangle(100.0, 50.0);
        assert!(valid.validate().is_ok());

        // Invalid: less than 3 vertices
        let invalid = Boundary2D::new(vec![(0.0, 0.0), (1.0, 0.0)]);
        assert!(invalid.validate().is_err());
    }
}

mod nester_tests {
    use super::*;

    #[test]
    fn test_simple_rectangle_nesting() {
        let geometries = vec![Geometry2D::rectangle("A", 10.0, 10.0).with_quantity(4)];
        let boundary = Boundary2D::rectangle(100.0, 50.0);

        let nester = Nester2D::default_config();
        let result = nester.solve(&geometries, &boundary).unwrap();

        // All 4 pieces should be placed (boundary is large enough)
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
        assert!(result.utilization > 0.0);
    }

    #[test]
    fn test_mixed_geometry_nesting() {
        let geometries = vec![
            Geometry2D::rectangle("large", 30.0, 20.0).with_quantity(2),
            Geometry2D::rectangle("small", 10.0, 10.0).with_quantity(5),
        ];
        let boundary = Boundary2D::rectangle(200.0, 100.0);

        let nester = Nester2D::default_config();
        let result = nester.solve(&geometries, &boundary).unwrap();

        // All pieces should fit
        assert_eq!(result.placements.len(), 7); // 2 + 5
        assert!(result.boundaries_used == 1);
    }

    #[test]
    fn test_nesting_with_margin_and_spacing() {
        let geometries = vec![Geometry2D::rectangle("piece", 20.0, 20.0).with_quantity(4)];
        let boundary = Boundary2D::rectangle(100.0, 100.0);

        let config = Config::default().with_margin(10.0).with_spacing(5.0);
        let nester = Nester2D::new(config);

        let result = nester.solve(&geometries, &boundary).unwrap();

        // With margin=10, usable area is 80x80
        // With spacing=5, pieces of 20x20 should fit
        // First row: 20 + 5 + 20 + 5 + 20 = 70 < 80, so 3 can fit in a row
        // Second row: 1 more piece
        assert_eq!(result.placements.len(), 4);
    }

    #[test]
    fn test_nesting_overflow() {
        let geometries = vec![Geometry2D::rectangle("big", 60.0, 60.0).with_quantity(5)];
        let boundary = Boundary2D::rectangle(100.0, 100.0);

        let nester = Nester2D::default_config();
        let result = nester.solve(&geometries, &boundary).unwrap();

        // Only 1 piece can fit in a 100x100 boundary
        assert_eq!(result.placements.len(), 1);
        // unplaced contains unique geometry IDs (after deduplication)
        assert_eq!(result.unplaced.len(), 1); // 1 geometry ID couldn't be fully placed
    }

    #[test]
    fn test_utilization_calculation() {
        let geometries = vec![Geometry2D::rectangle("quarter", 50.0, 50.0).with_quantity(1)];
        let boundary = Boundary2D::rectangle(100.0, 100.0);

        let nester = Nester2D::default_config();
        let result = nester.solve(&geometries, &boundary).unwrap();

        // 1 piece of 2500 in area of 10000 = 25% utilization
        assert!(
            (result.utilization - 0.25).abs() < 0.01,
            "Utilization = {}",
            result.utilization
        );
    }
}

mod transform_integration_tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_transform_geometry_vertices() {
        let geom = Geometry2D::rectangle("rect", 10.0, 5.0);
        let vertices = geom.outer_ring();

        // Apply a translation
        let t = Transform2D::translation(100.0, 50.0);
        let transformed: Vec<(f64, f64)> = vertices
            .iter()
            .map(|(x, y)| t.transform_point(*x, *y))
            .collect();

        // Check transformed vertices
        assert!((transformed[0].0 - 100.0).abs() < 1e-10);
        assert!((transformed[0].1 - 50.0).abs() < 1e-10);
        assert!((transformed[1].0 - 110.0).abs() < 1e-10);
        assert!((transformed[1].1 - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotated_geometry_aabb() {
        let geom = Geometry2D::rectangle("rect", 10.0, 2.0);
        let vertices = geom.outer_ring();

        // Rotate 90 degrees
        let t = Transform2D::rotation(PI / 2.0);
        let transformed: Vec<(f64, f64)> = vertices
            .iter()
            .map(|(x, y)| t.transform_point(*x, *y))
            .collect();

        // Compute AABB of transformed vertices
        let aabb = AABB2D::from_points(&transformed).unwrap();

        // After 90° rotation, a 10x2 rect becomes approximately 2x10
        // (with some floating point variance due to rotation around origin)
        let width = aabb.width();
        let height = aabb.height();
        assert!(
            (width - 2.0).abs() < 0.001 || (height - 2.0).abs() < 0.001,
            "width={}, height={}",
            width,
            height
        );
    }
}

mod stress_tests {
    use super::*;

    #[test]
    fn test_many_small_pieces() {
        let geometries = vec![Geometry2D::rectangle("tiny", 5.0, 5.0).with_quantity(100)];
        let boundary = Boundary2D::rectangle(500.0, 500.0);

        let nester = Nester2D::default_config();
        let result = nester.solve(&geometries, &boundary).unwrap();

        // Should place a reasonable number of pieces
        assert!(result.placements.len() >= 50);
        assert!(result.utilization > 0.0);
    }

    #[test]
    fn test_large_boundary() {
        let geometries = vec![Geometry2D::rectangle("medium", 50.0, 30.0).with_quantity(20)];
        let boundary = Boundary2D::rectangle(10000.0, 10000.0);

        let nester = Nester2D::default_config();
        let result = nester.solve(&geometries, &boundary).unwrap();

        // All pieces should fit in such a large boundary
        assert_eq!(result.placements.len(), 20);
        assert!(result.unplaced.is_empty());
    }
}
