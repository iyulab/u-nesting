//! Integration tests for u-nesting-core.

use u_nesting_core::ga::{GaConfig, PermutationChromosome};
use u_nesting_core::geometry::{Orientation3D, RotationConstraint};
use u_nesting_core::placement::{Placement, PlacementStats};
use u_nesting_core::result::SolveResult;
use u_nesting_core::transform::{Transform2D, Transform3D, AABB2D, AABB3D};

mod transform_tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_transform2d_composition() {
        // t1.then(t2) applies t2 first, then t1 (matrix multiplication order)
        // So: rotation first, then translation
        let t1 = Transform2D::translation(10.0, 0.0);
        let t2 = Transform2D::rotation(PI / 2.0);

        let composed = t1.then(&t2);
        let (x, y) = composed.transform_point(0.0, 0.0);

        // Point (0,0) -> rotation 90° -> (0,0) -> translate (10,0) -> (10,0)
        assert!((x - 10.0).abs() < 1e-10, "x = {}", x);
        assert!((y - 0.0).abs() < 1e-10, "y = {}", y);
    }

    #[test]
    fn test_transform2d_inverse_composition() {
        let t = Transform2D::new(5.0, 10.0, PI / 6.0);
        let inv = t.inverse();

        // T * T^-1 should be identity
        let identity = t.then(&inv);
        assert!(identity.is_identity(1e-10));

        // T^-1 * T should also be identity
        let identity2 = inv.then(&t);
        assert!(identity2.is_identity(1e-10));
    }

    #[test]
    fn test_transform3d_translation() {
        let t: Transform3D<f64> = Transform3D::translation(1.0, 2.0, 3.0);
        let (x, y, z) = t.transform_point(0.0, 0.0, 0.0);

        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
        assert!((z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform3d_rotation_x() {
        // Rotate around X axis by 90 degrees
        let t: Transform3D<f64> = Transform3D::rotation(PI / 2.0, 0.0, 0.0);
        let (x, y, z) = t.transform_point(0.0, 1.0, 0.0);

        // (0, 1, 0) rotated 90° around X should give (0, 0, 1)
        assert!((x - 0.0).abs() < 1e-10, "x = {}", x);
        assert!((y - 0.0).abs() < 1e-10, "y = {}", y);
        assert!((z - 1.0).abs() < 1e-10, "z = {}", z);
    }
}

mod aabb_tests {
    use super::*;

    #[test]
    fn test_aabb2d_operations() {
        let a: AABB2D<f64> = AABB2D::new(0.0, 0.0, 10.0, 10.0);
        let b: AABB2D<f64> = AABB2D::new(5.0, 5.0, 15.0, 15.0);

        // Test intersection
        assert!(a.intersects(&b));
        let intersection = a.intersection(&b).unwrap();
        assert!((intersection.min_x - 5.0).abs() < 1e-10);
        assert!((intersection.max_x - 10.0).abs() < 1e-10);

        // Test union
        let union = a.union(&b);
        assert!((union.min_x - 0.0).abs() < 1e-10);
        assert!((union.max_x - 15.0).abs() < 1e-10);

        // Test area
        assert!((a.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb2d_no_intersection() {
        let a: AABB2D<f64> = AABB2D::new(0.0, 0.0, 5.0, 5.0);
        let b: AABB2D<f64> = AABB2D::new(10.0, 10.0, 15.0, 15.0);

        assert!(!a.intersects(&b));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_aabb3d_operations() {
        let a: AABB3D<f64> = AABB3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b: AABB3D<f64> = AABB3D::new(5.0, 5.0, 5.0, 15.0, 15.0, 15.0);

        assert!(a.intersects(&b));
        assert!((a.volume() - 1000.0).abs() < 1e-10);

        let union = a.union(&b);
        assert!((union.volume() - 3375.0).abs() < 1e-10); // 15^3 = 3375
    }

    #[test]
    fn test_aabb_from_points() {
        let points: Vec<(f64, f64)> = vec![(1.0, 2.0), (5.0, 3.0), (2.0, 8.0), (7.0, 1.0)];

        let aabb = AABB2D::from_points(&points).unwrap();
        assert!((aabb.min_x - 1.0).abs() < 1e-10);
        assert!((aabb.min_y - 1.0).abs() < 1e-10);
        assert!((aabb.max_x - 7.0).abs() < 1e-10);
        assert!((aabb.max_y - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_expand() {
        let aabb: AABB2D<f64> = AABB2D::new(5.0, 5.0, 10.0, 10.0);
        let expanded = aabb.expand(2.0);

        assert!((expanded.min_x - 3.0).abs() < 1e-10);
        assert!((expanded.min_y - 3.0).abs() < 1e-10);
        assert!((expanded.max_x - 12.0).abs() < 1e-10);
        assert!((expanded.max_y - 12.0).abs() < 1e-10);
    }
}

mod placement_tests {
    use super::*;

    #[test]
    fn test_placement_transform_roundtrip() {
        let original = Placement::new_2d("test".to_string(), 0, 15.0_f64, 25.0, 0.785);
        let transform = original.to_transform_2d();
        let recovered = Placement::from_transform_2d("test".to_string(), 0, &transform);

        assert!((original.x() - recovered.x()).abs() < 1e-10);
        assert!((original.y() - recovered.y()).abs() < 1e-10);
        assert!((original.angle() - recovered.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_placement_3d_transform_roundtrip() {
        let original = Placement::new_3d("box".to_string(), 1, 10.0_f64, 20.0, 30.0, 0.1, 0.2, 0.3);
        let transform = original.to_transform_3d();
        let recovered = Placement::from_transform_3d("box".to_string(), 1, &transform);

        assert!((original.x() - recovered.x()).abs() < 1e-10);
        assert!((original.y() - recovered.y()).abs() < 1e-10);
        assert!((original.z().unwrap() - recovered.z().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_placement_stats_computation() {
        let placements = vec![
            Placement::new_2d("a".to_string(), 0, 0.0, 0.0, 0.0)
                .with_rotation_index(0)
                .with_boundary(0),
            Placement::new_2d("a".to_string(), 1, 10.0, 0.0, 0.0)
                .with_rotation_index(1)
                .with_boundary(0)
                .with_mirrored(true),
            Placement::new_2d("b".to_string(), 0, 0.0, 10.0, 0.0)
                .with_rotation_index(0)
                .with_boundary(1),
            Placement::new_2d("c".to_string(), 0, 20.0, 0.0, 0.0)
                .with_rotation_index(2)
                .with_boundary(0)
                .with_mirrored(true),
        ];

        let stats = PlacementStats::from_placements(&placements);

        assert_eq!(stats.count, 4);
        assert_eq!(stats.mirrored_count, 2);
        assert_eq!(stats.rotation_distribution.get(&0), Some(&2));
        assert_eq!(stats.rotation_distribution.get(&1), Some(&1));
        assert_eq!(stats.rotation_distribution.get(&2), Some(&1));
        assert_eq!(stats.boundary_distribution.get(&0), Some(&3));
        assert_eq!(stats.boundary_distribution.get(&1), Some(&1));
    }
}

mod rotation_constraint_tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_rotation_none() {
        let constraint: RotationConstraint<f64> = RotationConstraint::None;
        assert!(constraint.is_fixed());
        let angles = constraint.angles();
        assert_eq!(angles.len(), 1);
        assert!((angles[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_free() {
        let constraint: RotationConstraint<f64> = RotationConstraint::Free;
        assert!(!constraint.is_fixed());
        let angles = constraint.angles();
        assert!(angles.is_empty()); // Empty means any angle
    }

    #[test]
    fn test_rotation_axis_aligned() {
        let constraint: RotationConstraint<f64> = RotationConstraint::axis_aligned();
        let angles = constraint.angles();

        assert_eq!(angles.len(), 4);
        assert!((angles[0] - 0.0).abs() < 1e-10);
        assert!((angles[1] - PI / 2.0).abs() < 1e-10);
        assert!((angles[2] - PI).abs() < 1e-10);
        assert!((angles[3] - 3.0 * PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_steps() {
        let constraint: RotationConstraint<f64> = RotationConstraint::steps(6);
        let angles = constraint.angles();

        assert_eq!(angles.len(), 6);
        let step = 2.0 * PI / 6.0;
        for (i, angle) in angles.iter().enumerate() {
            let expected = step * i as f64;
            assert!(
                (angle - expected).abs() < 1e-10,
                "angle[{}] = {}, expected {}",
                i,
                angle,
                expected
            );
        }
    }
}

mod orientation_3d_tests {
    use super::*;

    #[test]
    fn test_orientation_counts() {
        assert_eq!(Orientation3D::Fixed.count(), 1);
        assert_eq!(Orientation3D::AxisAligned.count(), 6);
        assert_eq!(Orientation3D::Orthogonal.count(), 24);
        assert_eq!(Orientation3D::Free.count(), usize::MAX);
    }

    #[test]
    fn test_orientation_is_fixed() {
        assert!(Orientation3D::Fixed.is_fixed());
        assert!(!Orientation3D::AxisAligned.is_fixed());
        assert!(!Orientation3D::Orthogonal.is_fixed());
        assert!(!Orientation3D::Free.is_fixed());
    }
}

mod ga_tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_ga_config_builder() {
        let config = GaConfig::new()
            .with_population_size(200)
            .with_max_generations(1000)
            .with_crossover_rate(0.9)
            .with_mutation_rate(0.1)
            .with_elite_count(10);

        assert_eq!(config.population_size, 200);
        assert_eq!(config.max_generations, 1000);
        assert!((config.crossover_rate - 0.9).abs() < 1e-10);
        assert!((config.mutation_rate - 0.1).abs() < 1e-10);
        assert_eq!(config.elite_count, 10);
    }

    #[test]
    fn test_ga_config_clamping() {
        let config = GaConfig::new()
            .with_crossover_rate(1.5)  // Should be clamped to 1.0
            .with_mutation_rate(-0.1); // Should be clamped to 0.0

        assert!((config.crossover_rate - 1.0).abs() < 1e-10);
        assert!((config.mutation_rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_chromosome_validity() {
        let mut rng = StdRng::seed_from_u64(42);
        let chromosome = PermutationChromosome::random_with_options(20, 4, &mut rng);

        // Check it's a valid permutation
        assert_eq!(chromosome.genes.len(), 20);
        let mut sorted = chromosome.genes.clone();
        sorted.sort();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());

        // Check rotations are valid
        assert_eq!(chromosome.rotations.len(), 20);
        for r in &chromosome.rotations {
            assert!(*r < 4);
        }
    }

    #[test]
    fn test_permutation_crossover_validity() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent1 = PermutationChromosome::random_with_options(15, 6, &mut rng);
        let parent2 = PermutationChromosome::random_with_options(15, 6, &mut rng);

        // Perform multiple crossovers to ensure robustness
        for _ in 0..100 {
            let child = parent1.order_crossover(&parent2, &mut rng);

            // Child must be a valid permutation
            let mut sorted = child.genes.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                (0..15).collect::<Vec<_>>(),
                "Crossover produced invalid permutation"
            );
        }
    }

    #[test]
    fn test_permutation_mutation_validity() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chromosome = PermutationChromosome::random_with_options(25, 8, &mut rng);

        // Perform multiple mutations
        for _ in 0..100 {
            chromosome.swap_mutate(&mut rng);

            let mut sorted = chromosome.genes.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                (0..25).collect::<Vec<_>>(),
                "Swap mutation produced invalid permutation"
            );
        }
    }

    #[test]
    fn test_permutation_inversion_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chromosome = PermutationChromosome::random_with_options(10, 2, &mut rng);

        for _ in 0..50 {
            chromosome.inversion_mutate(&mut rng);

            let mut sorted = chromosome.genes.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                (0..10).collect::<Vec<_>>(),
                "Inversion mutation produced invalid permutation"
            );
        }
    }
}

mod solve_result_tests {
    use super::*;

    #[test]
    fn test_solve_result_creation() {
        let placements = vec![
            Placement::new_2d("a".to_string(), 0, 0.0, 0.0, 0.0),
            Placement::new_2d("b".to_string(), 0, 10.0, 0.0, 0.0),
        ];

        let result: SolveResult<f64> = SolveResult {
            placements,
            boundaries_used: 1,
            utilization: 0.75,
            unplaced: vec![],
            computation_time_ms: 150,
            generations: Some(50),
            iterations: None,
            best_fitness: Some(0.95),
            fitness_history: Some(vec![0.5, 0.7, 0.85, 0.95]),
            strategy: Some("BLF+GA".to_string()),
            cancelled: false,
            target_reached: true,
            strip_stats: vec![],
            total_piece_area: 0.0,
            total_material_used: 0.0,
            total_requested: 2,
            used_bounding_box: [0.0, 0.0],
        };

        assert_eq!(result.placements.len(), 2);
        assert_eq!(result.boundaries_used, 1);
        assert!((result.utilization - 0.75).abs() < 1e-10);
        assert!(result.target_reached);
        assert!(!result.cancelled);
    }
}
