//! Cutting sequence optimization.
//!
//! Determines the optimal order to cut contours, minimizing non-cutting
//! (rapid) travel distance while respecting precedence constraints.
//!
//! # Algorithms
//!
//! 1. **Nearest Neighbor (NN)**: Greedy construction heuristic that always
//!    selects the closest uncut contour that doesn't violate precedence.
//! 2. **Constrained 2-opt**: Local search improvement that reverses
//!    sub-sequences only when no precedence constraints are violated.
//!
//! # References
//!
//! - Dewil et al. (2016), Section 4: "Construction heuristics"

use std::collections::{HashMap, HashSet};

use u_nesting_core::timing::Timer;

use crate::common_edge::CommonEdgeResult;
use crate::config::CuttingConfig;
use crate::contour::CutContour;
use crate::cost::{closest_point_on_polygon, point_distance};
use crate::hierarchy::CuttingDag;
use crate::pierce::{select_pierce, PierceSelection};

/// Result of sequence optimization.
#[derive(Debug, Clone)]
pub struct SequenceResult {
    /// Contour IDs in cutting order.
    pub order: Vec<usize>,
    /// Pierce selections for each contour (indexed by order position).
    pub pierce_selections: Vec<PierceSelection>,
    /// Total rapid (non-cutting) distance.
    pub total_rapid_distance: f64,
}

/// Optimizes the cutting sequence using Nearest Neighbor + constrained 2-opt.
pub fn optimize_sequence(
    contours: &[CutContour],
    dag: &CuttingDag,
    config: &CuttingConfig,
) -> SequenceResult {
    optimize_sequence_with_adjacency(contours, dag, config, None)
}

/// Optimizes the cutting sequence with optional common-edge adjacency bonus.
///
/// When `common_edges` is provided, the NN heuristic applies a distance
/// discount for contours sharing edges with the previously cut contour.
/// This encourages consecutive cutting of adjacent parts, reducing rapid
/// travel and enabling potential common-edge single-pass cutting.
pub fn optimize_sequence_with_adjacency(
    contours: &[CutContour],
    dag: &CuttingDag,
    config: &CuttingConfig,
    common_edges: Option<&CommonEdgeResult>,
) -> SequenceResult {
    if contours.is_empty() {
        return SequenceResult {
            order: Vec::new(),
            pierce_selections: Vec::new(),
            total_rapid_distance: 0.0,
        };
    }

    // Build adjacency map from common edges
    let adjacency = build_adjacency_map(common_edges);

    // Index contours by id once: the 2-opt inner loop evaluates the tour cost
    // O(n^2) times per pass, and a linear `contours.iter().find()` inside each
    // evaluation would make that O(n) more expensive (the dominant cost on the
    // reported freeze). An id -> &contour map makes each lookup O(1).
    let index = build_contour_index(contours);

    // Step 1: Nearest Neighbor construction with adjacency bonus
    let mut order = nearest_neighbor_with_adjacency(contours, dag, config, &adjacency, &index);

    // Step 2: 2-opt improvement (wall-clock bounded via config.time_limit_ms)
    if config.max_2opt_iterations > 0 {
        improve_2opt(&mut order, dag, config, &index);
    }

    // Step 3: Compute pierce selections for the final order
    let (pierce_selections, total_rapid) = compute_pierce_selections(&order, &index, config);

    SequenceResult {
        order,
        pierce_selections,
        total_rapid_distance: total_rapid,
    }
}

/// Builds a map of adjacent contour pairs from common edge results.
/// Returns: contour_id -> set of adjacent contour_ids with shared edge lengths.
fn build_adjacency_map(
    common_edges: Option<&CommonEdgeResult>,
) -> HashMap<usize, Vec<(usize, f64)>> {
    let mut adjacency: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

    if let Some(result) = common_edges {
        for edge in &result.common_edges {
            adjacency
                .entry(edge.contour_a)
                .or_default()
                .push((edge.contour_b, edge.overlap_length));
            adjacency
                .entry(edge.contour_b)
                .or_default()
                .push((edge.contour_a, edge.overlap_length));
        }
    }

    adjacency
}

/// Builds an `id -> &contour` lookup so per-move cost evaluation avoids a
/// linear scan over `contours`.
fn build_contour_index(contours: &[CutContour]) -> HashMap<usize, &CutContour> {
    contours.iter().map(|c| (c.id, c)).collect()
}

/// Nearest Neighbor construction heuristic with adjacency bonus.
///
/// Starts from the home position and greedily selects the closest uncut
/// contour whose prerequisites are all already cut. Contours sharing
/// common edges with the last-cut contour receive a distance discount.
fn nearest_neighbor_with_adjacency(
    contours: &[CutContour],
    dag: &CuttingDag,
    config: &CuttingConfig,
    adjacency: &HashMap<usize, Vec<(usize, f64)>>,
    index: &HashMap<usize, &CutContour>,
) -> Vec<usize> {
    let n = contours.len();
    let mut visited: HashSet<usize> = HashSet::with_capacity(n);
    let mut order = Vec::with_capacity(n);
    let mut current_pos = config.home_position;
    let mut last_id: Option<usize> = None;

    // Adjacency discount factor: 0.5 means adjacent contours appear
    // 50% closer than their actual distance.
    const ADJACENCY_DISCOUNT: f64 = 0.5;

    for _ in 0..n {
        // Find the nearest unvisited contour whose prerequisites are satisfied
        let mut best_idx = None;
        let mut best_score = f64::MAX;

        for contour in contours.iter() {
            if visited.contains(&contour.id) {
                continue;
            }

            // Check if all predecessors have been visited
            let predecessors = dag.predecessors(contour.id);
            let ready = predecessors.iter().all(|pred_id| visited.contains(pred_id));

            if !ready {
                continue;
            }

            // Compute distance to nearest point on this contour
            let dist = closest_point_on_polygon(&contour.vertices, current_pos)
                .map(|(pt, _, _)| point_distance(current_pos, pt))
                .unwrap_or(f64::MAX);

            // Apply adjacency bonus if sharing a common edge with last-cut contour
            let mut score = dist;
            if let Some(last) = last_id {
                if let Some(neighbors) = adjacency.get(&last) {
                    if neighbors.iter().any(|(adj_id, _)| *adj_id == contour.id) {
                        score *= ADJACENCY_DISCOUNT;
                    }
                }
            }

            if score < best_score {
                best_score = score;
                best_idx = Some(contour.id);
            }
        }

        if let Some(id) = best_idx {
            visited.insert(id);
            order.push(id);
            last_id = Some(id);

            // Update current position to the pierce point
            if let Some(contour) = index.get(&id) {
                let pierce = select_pierce(contour, current_pos, config);
                current_pos = pierce.end_point;
            }
        }
    }

    order
}

/// How many candidate moves to evaluate between wall-clock checks.
///
/// Reading the clock on every move would cross the JS `performance.now()`
/// boundary `O(n^2)` times per pass on WASM. Amortizing keeps the overrun
/// tiny (a partial `i`-sweep) while avoiding that overhead.
const TIME_CHECK_INTERVAL: u32 = 512;

/// Constrained 2-opt improvement.
///
/// Tries to reverse sub-sequences in the order to reduce total rapid distance.
/// Only accepts reversals that don't violate precedence constraints.
///
/// Bounded by `config.time_limit_ms` (0 = unlimited): the check lives inside
/// the innermost `for j` loop because a single `i`/`j` double-loop pass is
/// itself the multi-second cost on large inputs — a check only at the outer
/// `while` level would not fire until after that pass completed. On timeout we
/// return with `order` in a valid, accepted state (every non-improving or
/// precedence-violating reversal is undone immediately), i.e. the best
/// sequence found so far.
fn improve_2opt(
    order: &mut [usize],
    dag: &CuttingDag,
    config: &CuttingConfig,
    index: &HashMap<usize, &CutContour>,
) {
    let n = order.len();
    if n < 3 {
        return;
    }

    let timer = Timer::now();
    let time_limited = config.time_limit_ms > 0;
    let mut since_check: u32 = 0;

    let mut improved = true;
    let mut iterations = 0;
    let mut current_rapid = compute_pierce_selections(order, index, config).1;

    while improved && iterations < config.max_2opt_iterations {
        improved = false;
        iterations += 1;

        for i in 0..n - 1 {
            for j in (i + 2)..n {
                // Wall-clock guard (amortized). `order` is in a clean accepted
                // state here — before the tentative reverse below — so
                // returning now yields a valid best-so-far sequence.
                since_check += 1;
                if time_limited && since_check >= TIME_CHECK_INTERVAL {
                    since_check = 0;
                    if timer.elapsed_ms() >= config.time_limit_ms {
                        return;
                    }
                }

                // Try reversing the segment [i+1..=j]
                order[i + 1..=j].reverse();

                // Check if the new sequence is valid
                if dag.is_valid_sequence(order) {
                    let new_rapid = compute_pierce_selections(order, index, config).1;

                    if new_rapid < current_rapid - 1e-10 {
                        current_rapid = new_rapid;
                        improved = true;
                    } else {
                        order[i + 1..=j].reverse(); // Undo — not an improvement
                    }
                } else {
                    order[i + 1..=j].reverse(); // Undo — violates precedence
                }
            }
        }
    }
}

/// Computes pierce selections and total rapid distance for a given order.
///
/// `index` maps contour id to contour, so this is `O(n)` per call rather than
/// `O(n^2)` — critical because the 2-opt loop calls it `O(n^2)` times per pass.
fn compute_pierce_selections(
    order: &[usize],
    index: &HashMap<usize, &CutContour>,
    config: &CuttingConfig,
) -> (Vec<PierceSelection>, f64) {
    let mut selections = Vec::with_capacity(order.len());
    let mut total_rapid = 0.0;
    let mut current_pos = config.home_position;

    for &contour_id in order {
        let contour = match index.get(&contour_id) {
            Some(c) => c,
            None => continue,
        };

        let pierce = select_pierce(contour, current_pos, config);
        let rapid = point_distance(current_pos, pierce.point);
        total_rapid += rapid;
        current_pos = pierce.end_point;
        selections.push(pierce);
    }

    (selections, total_rapid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contour::ContourType;

    fn make_contour(id: usize, cx: f64, cy: f64, ct: ContourType) -> CutContour {
        CutContour {
            id,
            geometry_id: format!("part{}", id),
            instance: 0,
            contour_type: ct,
            vertices: vec![
                (cx - 5.0, cy - 5.0),
                (cx + 5.0, cy - 5.0),
                (cx + 5.0, cy + 5.0),
                (cx - 5.0, cy + 5.0),
            ],
            perimeter: 40.0,
            centroid: (cx, cy),
        }
    }

    #[test]
    fn test_single_contour() {
        let contours = vec![make_contour(0, 50.0, 50.0, ContourType::Exterior)];
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        let result = optimize_sequence(&contours, &dag, &config);
        assert_eq!(result.order.len(), 1);
        assert_eq!(result.order[0], 0);
    }

    #[test]
    fn test_nn_selects_nearest() {
        // Three parts at increasing distances from origin
        let contours = vec![
            make_contour(0, 100.0, 0.0, ContourType::Exterior),
            make_contour(1, 20.0, 0.0, ContourType::Exterior),
            make_contour(2, 60.0, 0.0, ContourType::Exterior),
        ];
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        let result = optimize_sequence(&contours, &dag, &config);
        // NN should visit: nearest first (1 at x=20), then 2 (x=60), then 0 (x=100)
        assert_eq!(result.order, vec![1, 2, 0]);
    }

    #[test]
    fn test_precedence_respected() {
        // Part with interior hole — hole must come first
        let contours = vec![
            CutContour {
                id: 0,
                geometry_id: "part1".to_string(),
                instance: 0,
                contour_type: ContourType::Exterior,
                vertices: vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
                perimeter: 80.0,
                centroid: (10.0, 10.0),
            },
            CutContour {
                id: 1,
                geometry_id: "part1".to_string(),
                instance: 0,
                contour_type: ContourType::Interior,
                vertices: vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
                perimeter: 40.0,
                centroid: (10.0, 10.0),
            },
        ];
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        let result = optimize_sequence(&contours, &dag, &config);
        // Interior (id=1) must come before Exterior (id=0)
        let pos_interior = result
            .order
            .iter()
            .position(|&id| id == 1)
            .expect("interior should be in order");
        let pos_exterior = result
            .order
            .iter()
            .position(|&id| id == 0)
            .expect("exterior should be in order");
        assert!(pos_interior < pos_exterior);
    }

    #[test]
    fn test_empty_contours() {
        let contours: Vec<CutContour> = Vec::new();
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        let result = optimize_sequence(&contours, &dag, &config);
        assert!(result.order.is_empty());
        assert_eq!(result.total_rapid_distance, 0.0);
    }

    #[test]
    fn test_nn_better_than_reverse() {
        // Parts laid out in a line — NN should find a good order
        let contours: Vec<CutContour> = (0..5)
            .map(|i| make_contour(i, 20.0 * i as f64 + 10.0, 10.0, ContourType::Exterior))
            .collect();
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        let result = optimize_sequence(&contours, &dag, &config);

        // Compute rapid distance for reverse order
        let reverse_order: Vec<usize> = (0..5).rev().collect();
        let index = build_contour_index(&contours);
        let (_, reverse_rapid) = compute_pierce_selections(&reverse_order, &index, &config);

        assert!(
            result.total_rapid_distance <= reverse_rapid + 1e-6,
            "NN rapid {} should be <= reverse rapid {}",
            result.total_rapid_distance,
            reverse_rapid
        );
    }

    #[test]
    fn test_adjacency_bonus_prefers_neighbor() {
        // Three contours: 0 at origin, 1 far away, 2 also far but adjacent to 0
        // Without adjacency: NN visits 0 → 1 or 0 → 2 based on distance
        // With adjacency (0↔2 share edge): should prefer 0 → 2 → 1
        let contours = vec![
            make_contour(0, 10.0, 10.0, ContourType::Exterior),
            make_contour(1, 80.0, 10.0, ContourType::Exterior),
            make_contour(2, 90.0, 10.0, ContourType::Exterior),
        ];
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        // Create fake common edge between contour 0 and 2
        let common_edges = CommonEdgeResult {
            common_edges: vec![crate::common_edge::CommonEdge {
                contour_a: 0,
                edge_a: 0,
                contour_b: 2,
                edge_b: 0,
                overlap_length: 10.0,
                midpoint: (50.0, 10.0),
            }],
            total_common_length: 10.0,
        };

        let result_with =
            optimize_sequence_with_adjacency(&contours, &dag, &config, Some(&common_edges));
        let result_without = optimize_sequence(&contours, &dag, &config);

        // Both should produce valid sequences
        assert_eq!(result_with.order.len(), 3);
        assert_eq!(result_without.order.len(), 3);

        // With adjacency, contour 2 should be visited right after contour 0
        // since they share an edge (adjacency discount makes it appear closer)
        if result_with.order[0] == 0 {
            assert_eq!(
                result_with.order[1], 2,
                "Adjacent contour 2 should follow contour 0"
            );
        }
    }

    #[test]
    fn test_2opt_respects_time_limit() {
        // Regression: cut-path 2-opt must be wall-clock bounded so it cannot
        // freeze the (browser main) thread on large, legitimate inputs
        // ("N identical brackets on a sheet"). See
        // ISSUE-20260707-unesting-optimize-cutting-path-unbounded-runtime.
        //
        // 1500 distinct squares keep `improved == true` across passes, so
        // without a time bound the O(n^4)-per-pass 2-opt runs for many
        // seconds/minutes. With a 100ms budget it must return promptly with a
        // valid, complete order.
        let contours: Vec<CutContour> = (0..1500)
            .map(|i| {
                let cx = (i % 40) as f64 * 20.0;
                let cy = (i / 40) as f64 * 20.0;
                make_contour(i, cx, cy, ContourType::Exterior)
            })
            .collect();
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default()
            .with_max_2opt_iterations(1000)
            .with_time_limit_ms(100);

        let start = std::time::Instant::now();
        let result = optimize_sequence(&contours, &dag, &config);
        let elapsed = start.elapsed();

        // Early termination must still yield a valid, complete sequence.
        assert_eq!(result.order.len(), 1500);
        // Generous ceiling: the unbounded path took >5.6s at n=500 and is far
        // worse at n=1500. 5s never flakes yet catches an unbounded regression.
        assert!(
            elapsed.as_secs() < 5,
            "2-opt must respect time_limit_ms; took {elapsed:?}"
        );
    }

    #[test]
    fn test_zero_time_limit_is_unlimited() {
        // time_limit_ms == 0 preserves the anytime/iteration-cap semantics
        // (unbounded by wall clock) for native batch callers.
        let contours: Vec<CutContour> = (0..5)
            .map(|i| make_contour(i, 20.0 * i as f64 + 10.0, 10.0, ContourType::Exterior))
            .collect();
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default().with_time_limit_ms(0);

        let result = optimize_sequence(&contours, &dag, &config);
        assert_eq!(result.order.len(), 5);
    }

    #[test]
    fn test_adjacency_with_no_common_edges() {
        // No common edges — should behave identically to optimize_sequence
        let contours = vec![
            make_contour(0, 10.0, 10.0, ContourType::Exterior),
            make_contour(1, 30.0, 10.0, ContourType::Exterior),
        ];
        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::default();

        let result_with = optimize_sequence_with_adjacency(&contours, &dag, &config, None);
        let result_without = optimize_sequence(&contours, &dag, &config);

        assert_eq!(result_with.order, result_without.order);
    }
}
