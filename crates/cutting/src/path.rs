//! Final cutting path assembly and public API.
//!
//! Combines contour extraction, hierarchy analysis, sequence optimization,
//! and pierce point selection into a complete cutting path.

use std::time::Instant;

use u_nesting_core::geometry::{Geometry, Geometry2DExt};
use u_nesting_core::SolveResult;

use crate::common_edge;
use crate::config::CuttingConfig;
use crate::contour::extract_contours;
use crate::cost::point_distance;
use crate::gtsp;
use crate::hierarchy::CuttingDag;
use crate::kerf;
use crate::result::{CutStep, CuttingPathResult};
use crate::sequence::optimize_sequence;

/// Optimizes the cutting path for a nesting solve result.
///
/// This is the main entry point for cutting path optimization. It:
/// 1. Extracts contours from the placed geometries
/// 2. Builds precedence constraints (holes before exteriors)
/// 3. Optimizes the cutting sequence (NN + 2-opt)
/// 4. Selects optimal pierce points
/// 5. Assembles the final cutting path
///
/// # Arguments
///
/// * `solve_result` - The nesting solve result with placements
/// * `geometries` - The original geometry definitions
/// * `config` - Cutting path optimization parameters
///
/// # Returns
///
/// A `CuttingPathResult` with the optimized cutting sequence.
///
/// # Example
///
/// ```rust,ignore
/// use u_nesting_cutting::{CuttingConfig, optimize_cutting_path};
/// use u_nesting::d2::{Geometry2D, Boundary2D, Nester2D};
/// use u_nesting::core::{Solver, Config};
///
/// let geometries = vec![Geometry2D::rectangle("R1", 10.0, 5.0).with_quantity(3)];
/// let boundary = Boundary2D::rectangle(100.0, 50.0);
/// let nester = Nester2D::new(Config::default());
/// let solve_result = nester.solve(&geometries, &boundary).unwrap();
///
/// let cutting_config = CuttingConfig::default();
/// let path = optimize_cutting_path(&solve_result, &geometries, &cutting_config);
/// println!("Rapid distance: {}", path.total_rapid_distance);
/// ```
pub fn optimize_cutting_path<G>(
    solve_result: &SolveResult<f64>,
    geometries: &[G],
    config: &CuttingConfig,
) -> CuttingPathResult
where
    G: Geometry2DExt<Scalar = f64> + Geometry<Scalar = f64>,
{
    let start = Instant::now();

    // Step 1: Extract contours
    let raw_contours = extract_contours(solve_result, geometries);

    if raw_contours.is_empty() {
        return CuttingPathResult::new();
    }

    // Step 1.5: Apply kerf compensation (if kerf_width > 0)
    let contours = if config.kerf_width > 0.0 {
        let kerf_results = kerf::apply_kerf_compensation(&raw_contours, config);
        kerf::filter_compensated(kerf_results)
    } else {
        raw_contours
    };

    if contours.is_empty() {
        return CuttingPathResult::new();
    }

    // Step 2: Detect common edges (informational)
    let _common_edges = common_edge::detect_common_edges(
        &contours,
        config.kerf_width + config.tolerance,
        0.1,
    );

    // Step 3: Build precedence DAG
    let dag = CuttingDag::build(&contours);

    // Step 4: Optimize sequence
    // Use GTSP solver when multiple pierce candidates are configured
    let mut result = CuttingPathResult::new();
    let mut current_pos = config.home_position;

    if config.pierce_candidates > 1 {
        // GTSP path: discretize → build instance → solve with precedence
        let clusters = gtsp::discretize_contours(&contours, config);
        let instance = gtsp::build_gtsp_instance(clusters, config.home_position);
        let solution = gtsp::solve_constrained(&instance, &dag, config.max_2opt_iterations);

        for (i, &global_idx) in solution.iter().enumerate() {
            let candidate = instance.candidate(global_idx);
            let contour = match contours.iter().find(|c| c.id == candidate.contour_id) {
                Some(c) => c,
                None => continue,
            };

            let rapid_dist = point_distance(current_pos, candidate.point);

            result.sequence.push(CutStep {
                contour_id: candidate.contour_id,
                geometry_id: contour.geometry_id.clone(),
                instance: contour.instance,
                contour_type: contour.contour_type,
                pierce_point: candidate.point,
                cut_direction: candidate.direction,
                rapid_from: if i == 0 { None } else { Some(current_pos) },
                rapid_distance: rapid_dist,
                cut_distance: contour.perimeter,
            });

            result.total_rapid_distance += rapid_dist;
            result.total_cut_distance += contour.perimeter;
            result.total_pierces += 1;
            current_pos = candidate.end_point;
        }
    } else {
        // Legacy path: NN + 2-opt with single pierce selection
        let seq_result = optimize_sequence(&contours, &dag, config);

        for (i, &contour_id) in seq_result.order.iter().enumerate() {
            let contour = match contours.iter().find(|c| c.id == contour_id) {
                Some(c) => c,
                None => continue,
            };

            let pierce = &seq_result.pierce_selections[i];
            let rapid_dist = point_distance(current_pos, pierce.point);

            result.sequence.push(CutStep {
                contour_id,
                geometry_id: contour.geometry_id.clone(),
                instance: contour.instance,
                contour_type: contour.contour_type,
                pierce_point: pierce.point,
                cut_direction: pierce.direction,
                rapid_from: if i == 0 { None } else { Some(current_pos) },
                rapid_distance: rapid_dist,
                cut_distance: contour.perimeter,
            });

            result.total_rapid_distance += rapid_dist;
            result.total_cut_distance += contour.perimeter;
            result.total_pierces += 1;
            current_pos = pierce.end_point;
        }
    }

    result.computation_time_ms = start.elapsed().as_millis() as u64;

    // Estimate time if speeds are configured
    if config.rapid_speed > 0.0 && config.cut_speed > 0.0 {
        let rapid_time = result.total_rapid_distance / config.rapid_speed;
        let cut_time = result.total_cut_distance / config.cut_speed;
        result.estimated_time_seconds = Some(rapid_time + cut_time);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_solve_result() {
        let solve_result: SolveResult<f64> = SolveResult::new();
        let geometries: Vec<DummyGeom> = Vec::new();
        let config = CuttingConfig::default();

        let result = optimize_cutting_path(&solve_result, &geometries, &config);
        assert!(result.sequence.is_empty());
        assert_eq!(result.total_pierces, 0);
    }

    // Dummy geometry for testing (since Geometry2D is in u-nesting-d2)
    #[derive(Clone)]
    struct DummyGeom;

    impl u_nesting_core::geometry::Geometry for DummyGeom {
        type Scalar = f64;
        fn id(&self) -> &u_nesting_core::GeometryId {
            unimplemented!()
        }
        fn quantity(&self) -> usize {
            1
        }
        fn measure(&self) -> f64 {
            0.0
        }
        fn aabb_vec(&self) -> (Vec<f64>, Vec<f64>) {
            (vec![0.0, 0.0], vec![0.0, 0.0])
        }
        fn centroid(&self) -> Vec<f64> {
            vec![0.0, 0.0]
        }
        fn validate(&self) -> u_nesting_core::Result<()> {
            Ok(())
        }
        fn rotation_constraint(&self) -> &u_nesting_core::RotationConstraint<f64> {
            &u_nesting_core::RotationConstraint::None
        }
    }

    impl u_nesting_core::geometry::Geometry2DExt for DummyGeom {
        fn aabb_2d(&self) -> u_nesting_core::transform::AABB2D<f64> {
            u_nesting_core::transform::AABB2D::new(0.0, 0.0, 0.0, 0.0)
        }
        fn outer_ring(&self) -> &[(f64, f64)] {
            &[]
        }
        fn holes(&self) -> &[Vec<(f64, f64)>] {
            &[]
        }
        fn is_convex(&self) -> bool {
            true
        }
        fn convex_hull(&self) -> Vec<(f64, f64)> {
            Vec::new()
        }
        fn perimeter(&self) -> f64 {
            0.0
        }
    }
}
